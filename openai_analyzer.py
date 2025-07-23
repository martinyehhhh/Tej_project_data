#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API 公告分析器
針對 sbj_pu11 表中 RULC IN (1,11) 的公告進行結構化分析
生成摘要、when、how_much、who_what 四種 CSV 格式解析
"""

import openai
import pandas as pd
import json
import logging
import argparse
from datetime import datetime
from create_mysql_db import MySQLHandler
import configparser
import tiktoken
import time

class OpenAIAnalyzer:
    def __init__(self, config_file='config.ini', test_mode=False):
        self.config_file = config_file
        self.test_mode = test_mode
        self.load_config()
        self.setup_openai()
        
        # 限速處理相關變數
        self.rate_limit_count = 0  # 累計限速次數
        self.consecutive_success = 0  # 連續成功次數
        self.max_rate_limit_attempts = 4  # 最大重試次數（第4次限速時停止程式）
        self.base_wait_time = 60  # 基礎等待時間（秒）
        self.reset_threshold = 5  # 連續成功5次後重置限速計數
        
    def load_config(self):
        """載入設定檔"""
        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')
        self.openai_api_key = config.get('openai', 'api_key')
        
    def setup_openai(self):
        """設定 OpenAI API"""
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        # 初始化 tiktoken 編碼器
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def estimate_tokens(self, text):
        """估算文本的 token 數量"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # 如果編碼失敗，使用簡單的字元數估算（大約 4 字元 = 1 token）
            logging.warning(f"Token 計算失敗，使用字元數估算: {e}")
            return len(text) // 4
    
    def select_model(self, content, analysis_type="general"):
        """根據內容長度動態選擇最適合的模型
        
        Args:
            content: 要分析的內容
            analysis_type: 分析類型，用於決定輸出長度
            
        Returns:
            tuple: (model_name, max_tokens)
        """
        # 計算輸入 token 數
        input_tokens = self.estimate_tokens(content)
        
        # 根據分析類型設定預期輸出 token 數
        output_token_map = {
            "summary": 300,      # 摘要較短
            "when": 600,         # 時間資訊中等
            "how_much": 800,     # 數量金額較長
            "who_what": 800,     # 人物關係較長
            "general": 600       # 一般情況
        }
        
        expected_output_tokens = output_token_map.get(analysis_type, 600)
        
        # 總需求 token 數（輸入 + 輸出 + 系統訊息緩衝）
        total_tokens_needed = input_tokens + expected_output_tokens + 200
        
        # 根據 token 需求選擇模型
        if total_tokens_needed <= 15000:  # 留一些緩衝空間
            model = "gpt-3.5-turbo"
            max_context = 16000
            logging.info(f"選擇 gpt-3.5-turbo (預估需求: {total_tokens_needed} tokens)")
        elif total_tokens_needed <= 120000:  # 128k 的緩衝
            model = "gpt-4-turbo"
            max_context = 128000
            logging.info(f"選擇 gpt-4-turbo (預估需求: {total_tokens_needed} tokens)")
        else:
            model = "gpt-4o-mini"  # 使用 OpenAI 實際可用的大容量模型
            max_context = 128000   # gpt-4o-mini 實際上下文長度
            logging.info(f"選擇 gpt-4o-mini (預估需求: {total_tokens_needed} tokens)")
            
        return model, min(expected_output_tokens, 4000)  # 限制最大輸出長度
    
    def handle_rate_limit(self):
        """處理限速情況的智能等待機制"""
        self.rate_limit_count += 1
        self.consecutive_success = 0  # 重置連續成功計數
        
        # 檢查是否達到最大重試次數
        if self.rate_limit_count >= self.max_rate_limit_attempts:
            error_msg = f"已達到最大限速重試次數 ({self.max_rate_limit_attempts} 次)，程式將停止執行。"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        # 計算等待時間（指數退避：第1次60秒，第2次120秒，第3次240秒...）
        wait_time = self.base_wait_time * (2 ** (self.rate_limit_count - 1))
        wait_minutes = wait_time / 60
        
        logging.warning(f"遇到限速 (第 {self.rate_limit_count} 次)，將等待 {wait_minutes:.1f} 分鐘後重試...")
        logging.info(f"預計恢復時間: {(datetime.now() + pd.Timedelta(seconds=wait_time)).strftime('%H:%M:%S')}")
        
        # 分段等待，每30秒顯示一次剩餘時間
        remaining_time = wait_time
        while remaining_time > 0:
            sleep_duration = min(30, remaining_time)
            time.sleep(sleep_duration)
            remaining_time -= sleep_duration
            
            if remaining_time > 0:
                remaining_minutes = remaining_time / 60
                logging.info(f"還需等待 {remaining_minutes:.1f} 分鐘...")
        
        logging.info("等待結束，準備重新嘗試...")
    
    def record_success(self):
        """記錄成功執行，用於重置限速計數"""
        self.consecutive_success += 1
        
        # 如果連續成功達到閾值，重置限速計數
        if self.consecutive_success >= self.reset_threshold:
            if self.rate_limit_count > 0:
                logging.info(f"連續成功 {self.reset_threshold} 次，重置限速計數 (之前: {self.rate_limit_count} 次)")
                self.rate_limit_count = 0
            self.consecutive_success = 0
    
    def call_openai_with_retry(self, model, messages, max_tokens, temperature, analysis_type="未知"):
        """帶有限速重試機制的 OpenAI API 呼叫"""
        max_attempts = 3  # 每次分析的最大重試次數
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # 成功執行，記錄成功
                self.record_success()
                return response
                
            except openai.RateLimitError as e:
                logging.error(f"{analysis_type} 分析遇到限速錯誤: {e}")
                
                if attempt < max_attempts - 1:
                    # 還有重試機會，處理限速
                    self.handle_rate_limit()
                    logging.info(f"重試 {analysis_type} 分析 (第 {attempt + 2} 次嘗試)")
                else:
                    # 已達到最大重試次數
                    logging.error(f"{analysis_type} 分析達到最大重試次數，跳過此次分析")
                    raise e
                    
            except Exception as e:
                # 其他錯誤直接拋出
                logging.error(f"{analysis_type} 分析發生錯誤: {e}")
                raise e
        
        # 理論上不會到達這裡
        raise Exception(f"{analysis_type} 分析失敗")
    
    def log_openai_conversation(self, analysis_type, prompt, response, output_dir, file_prefix, model_used=None):
        """記錄 OpenAI 對話內容到檔案"""
        if not self.test_mode:
            return
            
        log_content = f"""=== OpenAI {analysis_type.upper()} 分析對話記錄 ===

【使用模型】
{model_used or '未指定'}

【發送給 OpenAI 的 Prompt】
{prompt}

【OpenAI 的回應】
{response}

【分析時間】
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
"""
        
        with open(f"{output_dir}/{file_prefix}_{analysis_type}_openai_log.txt", 'w', encoding='utf-8') as f:
            f.write(log_content)
        
    def get_cl1_announcements(self, mysql_handler, limit=None):
        """從資料庫取得 RULC=1,11 的公告，並對應 tej_pu11_1 表的 txt 內容"""
        if not mysql_handler.connection:
            logging.error("MySQL 未連接")
            return None
            
        try:
            cursor = mysql_handler.connection.cursor(dictionary=True)
            # 先查詢 sbj_pu11 中 rulc in (1,11) 且尚未處理的公告，包含所有需要的欄位
            sql = """
            SELECT id, ban, code, name, d_reals, hr_reals, od, cl, rulc
            FROM sbj_pu11 
            WHERE rulc IN (1, 11)
            AND openai_processed = 0 
            ORDER BY d_reals DESC, hr_reals DESC
            """
            if limit:
                sql += f" LIMIT {limit}"
                
            cursor.execute(sql)
            announcements = cursor.fetchall()
            
            # 對每個公告查詢對應的 tej_pu11_1 內容並合併
            results = []
            for announcement in announcements:
                content_sql = """
                SELECT txt 
                FROM tej_pu11_1 
                WHERE code = %s 
                AND d_reals = %s 
                AND hr_reals = %s 
                AND od = %s 
                AND txt IS NOT NULL
                ORDER BY id
                """
                cursor.execute(content_sql, (
                    announcement['code'], 
                    announcement['d_reals'], 
                    announcement['hr_reals'], 
                    announcement['od']
                ))
                content_rows = cursor.fetchall()
                
                if content_rows:
                    # 將所有 txt 內容合併
                    combined_content = '\n'.join([row['txt'] for row in content_rows if row['txt']])
                    announcement['content'] = combined_content
                    results.append(announcement)
            
            logging.info(f"找到 {len(results)} 筆 RULC IN (1,11) 且尚未處理的公告有對應的 tej_pu11_1 內容")
            return results
            
        except Exception as e:
            logging.error(f"查詢 RULC IN (1,11) 公告失敗: {e}")
            return None

    def analyze_summary(self, content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir=None, file_prefix=None):
        """生成摘要分析"""
        # 動態選擇模型
        model, max_tokens = self.select_model(content, "summary")
        
        prompt = f"""
請分析以下證交所公告內容，提供簡潔的摘要（50-100字）：

【公告基本資訊】
公告ID: {ann_id}
統一編號(BAN): {ban}
公司代碼: {code}
公司名稱: {name}
發言日期(D_REALS): {d_reals}
發言時間(HR_REALS): {hr_reals}
公告序號(OD): {od}
法規條款(RULC): {rulc}

【公告內容】
{content}

請以繁體中文回應，格式為純文字摘要。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.call_openai_with_retry(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                analysis_type="摘要"
            )
            result = response.choices[0].message.content.strip()
            
            # 記錄對話內容（如果是測試模式）
            if self.test_mode and output_dir and file_prefix:
                self.log_openai_conversation("summary", prompt, result, output_dir, file_prefix, model)
                
            return result
        except Exception as e:
            logging.error(f"摘要分析失敗: {e}")
            return "分析失敗"

    def analyze_when(self, content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir=None, file_prefix=None):
        """分析時間相關資訊"""
        # 動態選擇模型
        model, max_tokens = self.select_model(content, "when")
        
        prompt = f"""
請分析以下證交所公告內容，提取所有時間相關資訊，以 CSV 格式輸出。

【公告基本資訊】
公告ID: {ann_id}
統一編號(BAN): {ban}
公司代碼: {code}
公司名稱: {name}
發言日期(D_REALS): {d_reals}
發言時間(HR_REALS): {hr_reals}
公告序號(OD): {od}
法規條款(RULC): {rulc}

【公告內容】
{content}

重要注意事項：
1. 請保持原文中的時間格式，不要進行任何轉換
2. 如果是民國年，請維持民國年格式（如：113/07/09）
3. 如果是西元年，請維持西元年格式（如：2024/07/09）
4. 時間格式請完全依照原文，不要轉換成其他格式

請參考以下格式輸出 CSV：
項目說明,日期時間
發言日期,113/07/09
發言時間,21:34:40
事實發生日,113/07/09
交易期間（起始）,113/07/08
交易期間（結束）,113/07/09

請只輸出 CSV 格式，不要其他說明文字。如果某個時間資訊不存在，請省略該行。
維持原文的日期格式，不要進行民國年與西元年的轉換。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.call_openai_with_retry(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                analysis_type="時間"
            )
            result = response.choices[0].message.content.strip()
            
            # 記錄對話內容（如果是測試模式）
            if self.test_mode and output_dir and file_prefix:
                self.log_openai_conversation("when", prompt, result, output_dir, file_prefix, model)
                
            return result
        except Exception as e:
            logging.error(f"when 分析失敗: {e}")
            return "項目說明,日期時間\n分析失敗,N/A"

    def analyze_how_much(self, content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir=None, file_prefix=None):
        """分析數量金額相關資訊"""
        # 動態選擇模型
        model, max_tokens = self.select_model(content, "how_much")
        
        prompt = f"""
請分析以下證交所公告內容，提取所有數量、金額、比率相關資訊，以 CSV 格式輸出。

【公告基本資訊】
公告ID: {ann_id}
統一編號(BAN): {ban}
公司代碼: {code}
公司名稱: {name}
發言日期(D_REALS): {d_reals}
發言時間(HR_REALS): {hr_reals}
公告序號(OD): {od}
法規條款(RULC): {rulc}

【公告內容】
{content}

重要注意事項：
1. 請保持原文中的幣別格式，不要進行任何轉換或翻譯
2. 如果原文是「美元」，請保持「美元」，不要轉換成「USD」
3. 如果原文是「EUR」，請保持「EUR」，不要轉換成「歐元」
4. 如果原文是「新台幣」，請保持「新台幣」，不要轉換成「TWD」
5. 完全依照原文的幣別表達方式

請參考以下格式輸出 CSV：
類別,項目說明,數值（原始）,單位,幣別
數量,交易數量（本次收購）,9191782,股,N/A
金額,每單位價格（本次收購）,1.1,歐元/股,歐元
金額,交易總金額（本次收購）,10110960,歐元,歐元
比率,累積持股比例,92.21,%,N/A

注意事項：
1. 類別只能是：數量、金額、比率
2. 數值請移除千分位逗號
3. 幣別請完全依照原文表達，不要進行任何轉換
4. 如果無幣別概念，用 N/A

請只輸出 CSV 格式，不要其他說明文字。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.call_openai_with_retry(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                analysis_type="數量金額"
            )
            result = response.choices[0].message.content.strip()
            
            # 記錄對話內容（如果是測試模式）
            if self.test_mode and output_dir and file_prefix:
                self.log_openai_conversation("how_much", prompt, result, output_dir, file_prefix, model)
                
            return result
        except Exception as e:
            logging.error(f"how_much 分析失敗: {e}")
            return "類別,項目說明,數值（原始）,單位,幣別\n分析失敗,N/A,N/A,N/A,N/A"

    def analyze_who_what(self, content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir=None, file_prefix=None):
        """分析人物關係相關資訊"""
        # 動態選擇模型
        model, max_tokens = self.select_model(content, "who_what")
        
        prompt = f"""
請分析以下證交所公告內容，提取所有相關人物、公司、標的物及其關係，以 CSV 格式輸出。

【公告基本資訊】
公告ID: {ann_id}
統一編號(BAN): {ban}
公司代碼: {code}
公司名稱: {name}
發言日期(D_REALS): {d_reals}
發言時間(HR_REALS): {hr_reals}
公告序號(OD): {od}
法規條款(RULC): {rulc}

【公告內容】
{content}

請參考以下格式輸出 CSV：
項目,名稱,說明／關係
標的物,Cimpor Global Holdings B.V.（JVC）,一般普通股，公司擬取得其 100% 股權
買方（交易主體）,Taiwan Cement (Dutch) Holdings B.V.,台灣水泥公司之子公司
賣方（交易相對人）,OYAK Capital Investments B.V.,非關係人，無從屬或關聯關係
買方與賣方關係,-,無關係（確認非關係人交易）
買方與標的物關係,-,原持有 40% 股權，現擬增持至 100%

常見項目類別：
- 標的物、買方（交易主體）、賣方（交易相對人）
- 投資標的、轉讓方、受讓方
- 合作對象、子公司、母公司
- 各種關係說明

請只輸出 CSV 格式，不要其他說明文字。
保持原文中的公司名稱和關係描述，不要進行翻譯或轉換。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.call_openai_with_retry(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                analysis_type="人物關係"
            )
            result = response.choices[0].message.content.strip()
            
            # 記錄對話內容（如果是測試模式）
            if self.test_mode and output_dir and file_prefix:
                self.log_openai_conversation("who_what", prompt, result, output_dir, file_prefix, model)
                
            return result
        except Exception as e:
            logging.error(f"who_what 分析失敗: {e}")
            return "項目,名稱,說明／關係\n分析失敗,N/A,N/A"

    def process_announcements(self, announcements, mysql_handler, output_dir="./analysis_output"):
        """處理公告列表並生成分析結果"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 合併檔案路徑
        when_file_path = f"{output_dir}/all_when_analysis.csv"
        how_much_file_path = f"{output_dir}/all_how_much_analysis.csv"
        who_what_file_path = f"{output_dir}/all_who_what_analysis.csv"
        summary_file_path = f"{output_dir}/all_summary_analysis.txt"
        
        # 檢查檔案是否已存在，決定是否需要寫入表頭
        when_header_needed = not os.path.exists(when_file_path)
        how_much_header_needed = not os.path.exists(how_much_file_path)
        who_what_header_needed = not os.path.exists(who_what_file_path)
        
        # 以附加模式開啟檔案
        when_file = open(when_file_path, 'a', encoding='utf-8')
        how_much_file = open(how_much_file_path, 'a', encoding='utf-8')
        who_what_file = open(who_what_file_path, 'a', encoding='utf-8')
        summary_file = open(summary_file_path, 'a', encoding='utf-8')
        
        try:
            # 如果需要，寫入表頭
            if when_header_needed:
                when_file.write("公告ID,BAN,公司代碼,公司名稱,D_REALS,HR_REALS,OD,RULC,項目說明,日期時間\n")
            if how_much_header_needed:
                how_much_file.write("公告ID,BAN,公司代碼,公司名稱,D_REALS,HR_REALS,OD,RULC,類別,項目說明,數值（原始）,單位,幣別\n")
            if who_what_header_needed:
                who_what_file.write("公告ID,BAN,公司代碼,公司名稱,D_REALS,HR_REALS,OD,RULC,項目,名稱,說明／關係\n")
            
            for i, announcement in enumerate(announcements):
                try:
                    ann_id = announcement['id']
                    ban = announcement['ban']
                    code = announcement['code']
                    name = announcement['name']
                    d_reals = announcement['d_reals']
                    hr_reals = announcement['hr_reals']
                    od = announcement['od']
                    rulc = announcement['rulc']
                    content = announcement['content']  # 使用從 tej_pu11_1 表取得的 txt 內容
                    
                    logging.info(f"正在分析公告 {i+1}/{len(announcements)}: {code} {name} (ID: {ann_id})")
                    start_time = datetime.now()
                    
                    # 建立檔案名稱前綴
                    file_prefix = f"{ann_id}_{code}_{d_reals}"
                    
                    # 檢查內容長度並記錄
                    content_length = len(content)
                    estimated_tokens = self.estimate_tokens(content)
                    logging.info(f"公告內容長度: {content_length} 字元")
                    logging.info(f"預估 token 數: {estimated_tokens}")
                    
                    if content_length > 6000:
                        logging.warning(f"公告內容較長，可能會超過 AI 模型限制 (長度: {content_length})")
                    if estimated_tokens > 15000:
                        logging.warning(f"預估 token 數較高: {estimated_tokens}，將使用高容量模型")
                    
                    # 先儲存送給 OpenAI 的原始內容（保留個別檔案）
                    with open(f"{output_dir}/{file_prefix}_original_content.txt", 'w', encoding='utf-8') as f:
                        f.write("=== 送給 OpenAI 的原始內容 ===\n\n")
                        f.write(f"公告ID: {ann_id}\n")
                        f.write(f"BAN: {ban}\n")
                        f.write(f"公司代碼: {code}\n") 
                        f.write(f"公司名稱: {name}\n")
                        f.write(f"D_REALS: {d_reals}\n")
                        f.write(f"HR_REALS: {hr_reals}\n")
                        f.write(f"OD: {od}\n")
                        f.write(f"RULC: {rulc}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(content)
                        f.write("\n\n" + "=" * 50)
                    
                    # 生成各類分析（傳遞測試模式參數）
                    logging.info(f"開始進行 4 種分析...")
                    
                    summary_start = datetime.now()
                    summary = self.analyze_summary(content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir, file_prefix)
                    summary_time = datetime.now() - summary_start
                    logging.info(f"摘要分析完成 ({summary_time.total_seconds():.2f}秒)")
                    
                    when_start = datetime.now()
                    when_csv = self.analyze_when(content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir, file_prefix)
                    when_time = datetime.now() - when_start
                    logging.info(f"時間分析完成 ({when_time.total_seconds():.2f}秒)")
                    
                    how_much_start = datetime.now()
                    how_much_csv = self.analyze_how_much(content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir, file_prefix)
                    how_much_time = datetime.now() - how_much_start
                    logging.info(f"數量金額分析完成 ({how_much_time.total_seconds():.2f}秒)")
                    
                    who_what_start = datetime.now()
                    who_what_csv = self.analyze_who_what(content, ann_id, ban, code, name, d_reals, hr_reals, od, rulc, output_dir, file_prefix)
                    who_what_time = datetime.now() - who_what_start
                    logging.info(f"人物關係分析完成 ({who_what_time.total_seconds():.2f}秒)")
                    
                    # 寫入摘要到合併檔案
                    summary_file.write(f"=== 公告 {ann_id} - {ban} - {code} {name} ({d_reals}) ===\n")
                    summary_file.write(f"BAN: {ban}, D_REALS: {d_reals}, HR_REALS: {hr_reals}, OD: {od}, RULC: {rulc}\n")
                    summary_file.write(f"{summary}\n\n")
                    summary_file.flush()  # 立即寫入檔案
                    
                    # 處理 when CSV - 加入公告資訊欄位
                    when_lines = when_csv.strip().split('\n')
                    if when_lines and when_lines[0].strip():
                        # 跳過 CSV 的表頭行，從第二行開始處理
                        for line in when_lines[1:]:
                            if line.strip():
                                when_file.write(f"{ann_id},{ban},{code},{name},{d_reals},{hr_reals},{od},{rulc},{line}\n")
                        when_file.flush()  # 立即寫入檔案
                    
                    # 處理 how_much CSV - 加入公告資訊欄位
                    how_much_lines = how_much_csv.strip().split('\n')
                    if how_much_lines and how_much_lines[0].strip():
                        # 跳過 CSV 的表頭行，從第二行開始處理
                        for line in how_much_lines[1:]:
                            if line.strip():
                                how_much_file.write(f"{ann_id},{ban},{code},{name},{d_reals},{hr_reals},{od},{rulc},{line}\n")
                        how_much_file.flush()  # 立即寫入檔案
                    
                    # 處理 who_what CSV - 加入公告資訊欄位
                    who_what_lines = who_what_csv.strip().split('\n')
                    if who_what_lines and who_what_lines[0].strip():
                        # 跳過 CSV 的表頭行，從第二行開始處理
                        for line in who_what_lines[1:]:
                            if line.strip():
                                who_what_file.write(f"{ann_id},{ban},{code},{name},{d_reals},{hr_reals},{od},{rulc},{line}\n")
                        who_what_file.flush()  # 立即寫入檔案
                    
                    # 更新資料庫中的處理狀態
                    if mysql_handler.update_openai_processed_status(ann_id, True):
                        processing_time = datetime.now() - start_time
                        logging.info(f"公告 {ann_id} 分析完成並已標記為已處理 (總耗時: {processing_time.total_seconds():.2f}秒)")
                    else:
                        logging.warning(f"公告 {ann_id} 分析完成但狀態更新失敗")
                    
                except Exception as e:
                    logging.error(f"處理公告 {announcement.get('id', 'unknown')} 失敗: {e}")
                    logging.exception("詳細錯誤資訊:")
                    continue
        
        finally:
            # 關閉所有檔案
            when_file.close()
            how_much_file.close()
            who_what_file.close()
            summary_file.close()

def main():
    parser = argparse.ArgumentParser(description="OpenAI 公告分析器")
    parser.add_argument('--config', default='config.ini', help='設定檔路徑')
    parser.add_argument('--limit', type=int, default=20, help='處理公告數量限制')
    parser.add_argument('--output-dir', default='./analysis_output', help='輸出目錄')
    parser.add_argument('--test-mode', action='store_true', help='測試模式：記錄所有與OpenAI的對話內容')
    parser.add_argument('--log-file', default=None, help='日誌檔案路徑（預設：自動生成時間戳記檔名）')
    args = parser.parse_args()
    
    # 設定日誌檔案路徑
    if args.log_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = f"./logs/openai_analyzer_{timestamp}.log"
    
    # 建立日誌目錄
    import os
    log_dir = os.path.dirname(args.log_file) or './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 設定日誌記錄器
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, encoding='utf-8'),  # 寫入檔案
            logging.StreamHandler()  # 同時顯示在終端
        ]
    )
    
    logging.info("=" * 60)
    logging.info("OpenAI 公告分析器開始執行")
    logging.info(f"執行參數: config={args.config}, limit={args.limit}, output_dir={args.output_dir}")
    logging.info(f"測試模式: {'啟用' if args.test_mode else '關閉'}")
    logging.info(f"限速處理機制: 啟用 (最大重試 {4} 次，連續成功 {5} 次重置)")
    logging.info(f"日誌檔案: {args.log_file}")
    logging.info("=" * 60)
    
    try:
        # 初始化分析器（傳入測試模式參數）
        analyzer = OpenAIAnalyzer(config_file=args.config, test_mode=args.test_mode)
        
        if args.test_mode:
            logging.info("啟用測試模式：將記錄所有與OpenAI的對話內容")
        
        # 連接資料庫
        mysql_handler = MySQLHandler(config_file=args.config)
        if not mysql_handler.connect():
            logging.error("連接 MySQL 失敗")
            return
            
        # 選擇資料庫
        if not mysql_handler.select_database():
            logging.error("選擇資料庫失敗")
            return
            
        # 取得 RULC IN (1,11) 的公告
        announcements = analyzer.get_cl1_announcements(mysql_handler, limit=args.limit)
        if not announcements:
            logging.error("未找到 RULC IN (1,11) 的公告")
            return
            
        logging.info(f"找到 {len(announcements)} 筆 RULC IN (1,11) 公告，開始分析...")
        
        # 處理公告
        start_time = datetime.now()
        analyzer.process_announcements(announcements, mysql_handler, output_dir=args.output_dir)
        end_time = datetime.now()
        
        mysql_handler.close()
        
        processing_time = end_time - start_time
        logging.info("=" * 60)
        logging.info("分析完成")
        logging.info(f"總處理時間: {processing_time}")
        logging.info(f"平均每筆公告處理時間: {processing_time / len(announcements)}")
        logging.info(f"限速統計: 總計 {analyzer.rate_limit_count} 次，連續成功 {analyzer.consecutive_success} 次")
        logging.info(f"日誌檔案已儲存至: {args.log_file}")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"程式執行失敗: {e}")
        logging.exception("詳細錯誤資訊:")
        raise

if __name__ == '__main__':
    main()
