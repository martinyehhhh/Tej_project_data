#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL 資料庫建立與資料匯入工具
用於 SBJ_PU11 重大訊息資料
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import argparse
import logging
from datetime import datetime
import sys
import configparser
import os


class MySQLHandler:
    """MySQL 資料庫處理類別"""
    
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = None
        self.connection = None
        self._load_config()
        
    def _load_config(self):
        """從設定檔載入資料庫連線資訊"""
        if not os.path.exists(self.config_file):
            logging.error(f"設定檔 {self.config_file} 不存在")
            raise FileNotFoundError(f"設定檔 {self.config_file} 不存在")

        self.config = configparser.ConfigParser()
        self.config.read(self.config_file, encoding='utf-8')

        # 驗證必要的設定項目
        required_keys = ['host', 'user', 'password', 'database']
        for key in required_keys:
            if not self.config.get('mysql', key, fallback=None):
                logging.error(f"設定檔中缺少必要項目: mysql.{key}")
                raise ValueError(f"設定檔中缺少必要項目: mysql.{key}")

        logging.info(f"已載入設定檔: {self.config_file}")
        
    def get_connection_params(self):
        """取得連線參數"""
        return {
            'host': self.config.get('mysql', 'host'),
            'port': self.config.getint('mysql', 'port', fallback=3306),
            'user': self.config.get('mysql', 'user'),
            'password': self.config.get('mysql', 'password'),
            'charset': self.config.get('mysql', 'charset', fallback='utf8mb4'),
            'connect_timeout': self.config.getint('options', 'connect_timeout', fallback=10),
            'autocommit': self.config.getboolean('options', 'autocommit', fallback=False)
        }
        
    def connect(self):
        """連接到 MySQL 資料庫"""
        try:
            params = self.get_connection_params()
            # 先嘗試連接到 MySQL 服務器（不指定資料庫）
            connect_params = params.copy()
            connect_params.pop('database', None)  # 移除資料庫參數
            
            self.connection = mysql.connector.connect(**connect_params)
            logging.info(f"成功連接到 MySQL 服務器 ({params['host']}:{params['port']})")
            return True
        except Error as e:
            logging.error(f"連接 MySQL 失敗: {e}")
            return False
    
    def create_database(self):
        """創建資料庫"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        try:
            database_name = self.config.get('mysql', 'database')
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {database_name}")
            logging.info(f"資料庫 {database_name} 創建成功")
            return True
        except Error as e:
            logging.error(f"創建資料庫失敗: {e}")
            return False
    
    def create_table(self):
        """創建 SBJ_PU11 資料表"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS sbj_pu11 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ban VARCHAR(8),
            code VARCHAR(7),
            name VARCHAR(8),
            d_reals INT,
            od INT,
            hr_reals INT,
            occur_d INT,
            bandayhr VARCHAR(24),
            rulb INT,
            erx VARCHAR(1),
            rulc INT,
            txtt TEXT,
            mkt VARCHAR(3),
            tt1 VARCHAR(2) COMMENT '主旨末尾 2 字',
            tt2 VARCHAR(4) COMMENT '主旨末尾 4 字',
            cl INT COMMENT '重大訊息分類(共 12 類，預設 99)',
            openai_processed TINYINT(1) DEFAULT 0 COMMENT '是否已經生成 OpenAI 報告 (0:未處理, 1:已處理)',
            openai_processed_at TIMESTAMP NULL COMMENT 'OpenAI 報告生成時間',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            logging.info("資料表 sbj_pu11 創建成功")
            return True
        except Error as e:
            logging.error(f"創建資料表失敗: {e}")
            return False
    
    def insert_dataframe(self, df: pd.DataFrame, batch_size: int = None):
        """將 DataFrame 插入資料庫"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        if df.empty:
            logging.warning("DataFrame 為空，沒有資料可插入")
            return False
        
        # 從設定檔取得批次大小
        if batch_size is None:
            batch_size = self.config.getint('options', 'batch_size', fallback=1000)
        
        columns = [
            'ban', 'code', 'name', 'd_reals', 'od', 'hr_reals', 
            'occur_d', 'bandayhr', 'rulb', 'erx', 'rulc', 'txtt', 'mkt',
            'tt1', 'tt2', 'cl'
        ]
        
        # 過濾存在的欄位
        available_columns = [col for col in columns if col.upper() in df.columns or col in df.columns]
        
        placeholders = ', '.join(['%s'] * len(available_columns))
        insert_sql = f"""
        INSERT INTO sbj_pu11 ({', '.join(available_columns)}) 
        VALUES ({placeholders})
        """
        
        try:
            cursor = self.connection.cursor()
            
            # 分批插入
            total_rows = len(df)
            inserted_rows = 0
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # 準備批次資料
                batch_data = []
                for _, row in batch_df.iterrows():
                    row_data = []
                    for col in available_columns:
                        # 統一欄位名稱大小寫
                        col_upper = col.upper()
                        if col_upper in row:
                            value = row[col_upper]
                        elif col in row:
                            value = row[col]
                        else:
                            value = None
                        
                        # 處理 NaN 值
                        if pd.isna(value):
                            value = None
                        row_data.append(value)
                    
                    batch_data.append(tuple(row_data))
                
                # 執行批次插入
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                
                inserted_rows += len(batch_data)
                logging.info(f"已插入 {inserted_rows}/{total_rows} 筆記錄")
            
            logging.info(f"成功插入 {inserted_rows} 筆記錄到資料庫")
            return True
            
        except Error as e:
            logging.error(f"插入資料失敗: {e}")
            self.connection.rollback()
            return False
    
    def query_data(self, limit: int = 10):
        """查詢資料"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(f"SELECT * FROM sbj_pu11 ORDER BY id DESC LIMIT {limit}")
            results = cursor.fetchall()
            return results
        except Error as e:
            logging.error(f"查詢資料失敗: {e}")
            return None
    
    def get_classification_stats(self):
        """取得分類統計"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT cl, COUNT(*) as count 
                FROM sbj_pu11 
                GROUP BY cl 
                ORDER BY cl
            """)
            results = cursor.fetchall()
            return results
        except Error as e:
            logging.error(f"查詢分類統計失敗: {e}")
            return None
    
    def create_tej_pu11_table(self):
        """創建 TEJ_PU11 資料表"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS tej_pu11_1 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ban VARCHAR(8) COMMENT '統編',
            code VARCHAR(7) COMMENT '公司碼',
            name VARCHAR(20) COMMENT '公司名稱',
            gdate INT COMMENT '抓檔日',
            hhmmss INT COMMENT '傳輸時分秒',
            date INT COMMENT '輸入日',
            od INT COMMENT '則次',
            hr_reals INT COMMENT '發言時間',
            file_nm VARCHAR(70) COMMENT '外部檔檔名',
            occur_d INT COMMENT '事實發生日',
            spoker VARCHAR(12) COMMENT '發言人',
            d_reals INT COMMENT '發言日期',
            keyin1 INT COMMENT '更改日期',
            key_hr INT COMMENT '異動時間',
            rula INT COMMENT '法條',
            rulb INT COMMENT '款',
            dbcl VARCHAR(9) COMMENT 'DB類',
            mkt VARCHAR(3) COMMENT '上市別',
            no VARCHAR(5) COMMENT '列次',
            txt VARCHAR(70) COMMENT '內容',
            hm_ann INT COMMENT '發言時間(小時)',
            cla VARCHAR(1) COMMENT 'DB類第一字元',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            logging.info("資料表 tej_pu11_1 創建成功")
            return True
        except Error as e:
            logging.error(f"創建資料表失敗: {e}")
            return False

    def insert_tej_pu11_dataframe(self, df: pd.DataFrame, batch_size: int = None):
        """將 TEJ_PU11 DataFrame 插入資料庫"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        if df.empty:
            logging.warning("DataFrame 為空，沒有資料可插入")
            return False
        
        # 從設定檔取得批次大小
        if batch_size is None:
            batch_size = self.config.getint('options', 'batch_size', fallback=1000)
        
        columns = [
            'ban', 'code', 'name', 'gdate', 'hhmmss', 'date', 'od', 'hr_reals',
            'file_nm', 'occur_d', 'spoker', 'd_reals', 'keyin1', 'key_hr',
            'rula', 'rulb', 'dbcl', 'mkt', 'no', 'txt', 'hm_ann', 'cla'
        ]
        
        # 過濾存在的欄位
        available_columns = [col for col in columns if col.upper() in df.columns or col in df.columns]
        
        placeholders = ', '.join(['%s'] * len(available_columns))
        insert_sql = f"""
        INSERT INTO tej_pu11_1 ({', '.join(available_columns)}) 
        VALUES ({placeholders})
        """
        
        try:
            cursor = self.connection.cursor()
            
            # 分批插入
            total_rows = len(df)
            inserted_rows = 0
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # 準備批次資料
                batch_data = []
                for _, row in batch_df.iterrows():
                    row_data = []
                    for col in available_columns:
                        # 統一欄位名稱大小寫
                        col_upper = col.upper()
                        if col_upper in row:
                            value = row[col_upper]
                        elif col in row:
                            value = row[col]
                        else:
                            value = None
                        
                        # 處理 NaN 值
                        if pd.isna(value):
                            value = None
                        row_data.append(value)
                    
                    batch_data.append(tuple(row_data))
                
                # 執行批次插入
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                
                inserted_rows += len(batch_data)
                logging.info(f"已插入 {inserted_rows}/{total_rows} 筆記錄")
            
            logging.info(f"成功插入 {inserted_rows} 筆記錄到 tej_pu11_1 資料表")
            return True
            
        except Error as e:
            logging.error(f"插入資料失敗: {e}")
            self.connection.rollback()
            return False

    def query_tej_pu11_data(self, limit: int = 10):
        """查詢 TEJ_PU11 資料"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(f"SELECT * FROM tej_pu11_1 ORDER BY id DESC LIMIT {limit}")
            results = cursor.fetchall()
            return results
        except Error as e:
            logging.error(f"查詢資料失敗: {e}")
            return None

    def get_tej_pu11_stats(self):
        """取得 TEJ_PU11 統計資料"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ban) as unique_companies,
                    COUNT(DISTINCT cla) as unique_cla,
                    MIN(gdate) as min_date,
                    MAX(gdate) as max_date
                FROM tej_pu11_1
            """)
            result = cursor.fetchone()
            return result
        except Error as e:
            logging.error(f"查詢統計失敗: {e}")
            return None
    
    def select_database(self):
        """選擇資料庫"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        try:
            database_name = self.config.get('mysql', 'database')
            cursor = self.connection.cursor()
            cursor.execute(f"USE {database_name}")
            logging.info(f"已選擇資料庫: {database_name}")
            return True
        except Error as e:
            logging.error(f"選擇資料庫失敗: {e}")
            return False
    
    def update_openai_processed_status(self, record_id, status=True):
        """更新 OpenAI 處理狀態"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        try:
            cursor = self.connection.cursor()
            sql = """
            UPDATE sbj_pu11 
            SET openai_processed = %s, openai_processed_at = NOW()
            WHERE id = %s
            """
            cursor.execute(sql, (1 if status else 0, record_id))
            self.connection.commit()
            logging.info(f"已更新記錄 {record_id} 的 OpenAI 處理狀態為: {status}")
            return True
        except Error as e:
            logging.error(f"更新 OpenAI 處理狀態失敗: {e}")
            return False
    
    def reset_all_openai_processed_status(self):
        """重置所有記錄的 OpenAI 處理狀態為未處理"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # 先查詢目前已處理的記錄數量
            cursor.execute("SELECT COUNT(*) as count FROM sbj_pu11 WHERE openai_processed = 1")
            result = cursor.fetchone()
            processed_count = result[0] if result else 0
            
            if processed_count == 0:
                logging.info("目前沒有已處理的記錄，無需重置")
                return True
            
            # 重置所有已處理記錄的狀態
            sql = """
            UPDATE sbj_pu11 
            SET openai_processed = 0, openai_processed_at = NULL
            WHERE openai_processed = 1
            """
            cursor.execute(sql)
            affected_rows = cursor.rowcount
            self.connection.commit()
            
            logging.info(f"成功重置 {affected_rows} 筆記錄的 OpenAI 處理狀態為未處理")
            return True
            
        except Error as e:
            logging.error(f"重置 OpenAI 處理狀態失敗: {e}")
            self.connection.rollback()
            return False
    
    def get_openai_processed_stats(self):
        """取得 OpenAI 處理狀態統計"""
        if not self.connection:
            logging.error("未連接到 MySQL")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    openai_processed,
                    COUNT(*) as count
                FROM sbj_pu11 
                GROUP BY openai_processed
                ORDER BY openai_processed
            """)
            results = cursor.fetchall()
            return results
        except Error as e:
            logging.error(f"查詢 OpenAI 處理狀態統計失敗: {e}")
            return None
    
    def close(self):
        """關閉資料庫連接"""
        if self.connection:
            self.connection.close()
            logging.info("MySQL 連接已關閉")


def main():
    parser = argparse.ArgumentParser(description="MySQL 資料庫建立與資料匯入工具")
    parser.add_argument('--config', default='config.ini', help='資料庫設定檔路徑')
    parser.add_argument('--csv-file', help='要匯入的 CSV 檔案路徑')
    parser.add_argument('--create-only', action='store_true', help='僅創建資料庫和資料表')
    parser.add_argument('--query', action='store_true', help='查詢最新資料')
    parser.add_argument('--stats', action='store_true', help='顯示分類統計')
    
    # TEJ_PU11 相關參數
    parser.add_argument('--table', choices=['sbj_pu11', 'tej_pu11_1'], default='sbj_pu11', 
                       help='指定操作的資料表')
    parser.add_argument('--create-tej', action='store_true', help='創建 TEJ_PU11 資料表')
    parser.add_argument('--query-tej', action='store_true', help='查詢 TEJ_PU11 最新資料')
    parser.add_argument('--stats-tej', action='store_true', help='顯示 TEJ_PU11 統計')
    
    # OpenAI 處理狀態相關參數
    parser.add_argument('--reset-openai-status', action='store_true', 
                       help='重置所有記錄的 OpenAI 處理狀態為未處理')
    parser.add_argument('--openai-stats', action='store_true', 
                       help='顯示 OpenAI 處理狀態統計')
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mysql_operations.log', encoding='utf-8')
        ]
    )
    
    try:
        # 創建 MySQL 處理器
        mysql_handler = MySQLHandler(config_file=args.config)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"設定檔錯誤: {e}")
        return
    
    try:
        # 連接資料庫
        if not mysql_handler.connect():
            return
        
        # 創建資料庫
        if not mysql_handler.create_database():
            return
        
        # 創建 SBJ_PU11 資料表
        if args.create_only:
            if mysql_handler.create_table():
                logging.info("SBJ_PU11 資料表創建完成")
            return
        
        # 創建 TEJ_PU11 資料表
        if args.create_tej:
            if mysql_handler.create_tej_pu11_table():
                logging.info("TEJ_PU11 資料表創建完成")
            return
        
        # 匯入 CSV 資料
        if args.csv_file:
            try:
                df = pd.read_csv(args.csv_file, encoding='utf-8-sig')
                logging.info(f"讀取 CSV 檔案: {args.csv_file}, 共 {len(df)} 筆記錄")
                
                if args.table == 'tej_pu11_1':
                    # 確保 TEJ_PU11 資料表存在
                    mysql_handler.create_tej_pu11_table()
                    if mysql_handler.insert_tej_pu11_dataframe(df):
                        logging.info("TEJ_PU11 資料匯入成功")
                    else:
                        logging.error("TEJ_PU11 資料匯入失敗")
                else:
                    # 預設為 SBJ_PU11
                    mysql_handler.create_table()
                    if mysql_handler.insert_dataframe(df):
                        logging.info("SBJ_PU11 資料匯入成功")
                    else:
                        logging.error("SBJ_PU11 資料匯入失敗")
                        
            except Exception as e:
                logging.error(f"讀取 CSV 檔案失敗: {e}")
        
        # 查詢 SBJ_PU11 資料
        if args.query:
            results = mysql_handler.query_data()
            if results:
                print("\nSBJ_PU11 最新 10 筆記錄:")
                for result in results:
                    print(f"ID: {result['id']}, BAN: {result['ban']}, "
                          f"CL: {result['cl']}, CODE: {result['code']}")
        
        # 查詢 TEJ_PU11 資料
        if args.query_tej:
            results = mysql_handler.query_tej_pu11_data()
            if results:
                print("\nTEJ_PU11 最新 10 筆記錄:")
                for result in results:
                    print(f"ID: {result['id']}, BAN: {result['ban']}, "
                          f"NAME: {result['name']}, CLA: {result['cla']}")
        
        # 顯示 SBJ_PU11 統計
        if args.stats:
            stats = mysql_handler.get_classification_stats()
            if stats:
                print("\nSBJ_PU11 分類統計:")
                for stat in stats:
                    print(f"分類 {stat['cl']}: {stat['count']} 筆")
        
        # 顯示 TEJ_PU11 統計
        if args.stats_tej:
            stats = mysql_handler.get_tej_pu11_stats()
            if stats:
                print("\nTEJ_PU11 統計資料:")
                print(f"總記錄數: {stats['total_records']}")
                print(f"公司數量: {stats['unique_companies']}")
                print(f"CLA 類別數: {stats['unique_cla']}")
                print(f"日期範圍: {stats['min_date']} - {stats['max_date']}")
        
        # 重置 OpenAI 處理狀態
        if args.reset_openai_status:
            print("\n正在重置所有記錄的 OpenAI 處理狀態...")
            if mysql_handler.reset_all_openai_processed_status():
                print("✓ OpenAI 處理狀態重置完成")
            else:
                print("✗ OpenAI 處理狀態重置失敗")
        
        # 顯示 OpenAI 處理狀態統計
        if args.openai_stats:
            stats = mysql_handler.get_openai_processed_stats()
            if stats:
                print("\nOpenAI 處理狀態統計:")
                for stat in stats:
                    status_text = "已處理" if stat['openai_processed'] == 1 else "未處理"
                    print(f"{status_text}: {stat['count']} 筆")
            else:
                print("無法取得 OpenAI 處理狀態統計")
                    
    finally:
        mysql_handler.close()


if __name__ == '__main__':
    main()
