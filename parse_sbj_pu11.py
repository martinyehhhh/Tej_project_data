#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBJ_PU11 文件結構化解析工具 (Big5 固定長度)
依 sbj_pu11mas.txt 定義，以 bytes 精準切割，再 decode
新增欄位：
  TT1: 主旨末尾 2 字
  TT2: 主旨末尾 4 字
  CL: 重大訊息分類(共 12 類，預設 99)
"""

import argparse
import logging
from datetime import datetime

import pandas as pd
from create_mysql_db import MySQLHandler


class SBJ_PU11_Parser:
    """SBJ_PU11 固定長度檔案解析器"""

    def __init__(self):
        # name, start_byte, length_bytes, type ('A'=alpha/text, 'I'=integer)
        self.field_defs = [
            ('BAN',       0,   8,   'A'),
            ('CODE',      8,   7,   'A'),
            ('NAME',      15,  8,   'A'),
            ('D_REALS',   23,  8,   'I'),
            ('OD',        31,  2,   'I'),
            ('HR_REALS',  33,  6,   'I'),
            ('OCCUR_D',   39,  8,   'I'),
            ('BANDAYHR',  47,  24,  'A'),
            ('RULB',      71,  3,   'I'),
            ('ERX',       74,  1,   'A'),
            ('RULC',      75,  2,   'I'),
            ('TXTT',      77,  210, 'A'),
            ('MKT',       287, 3,   'A'),
        ]
        self.total_length = max(s + l for _, s, l, _ in self.field_defs)

    def parse_file(self, infile: str, max_lines: int = None) -> pd.DataFrame:
        """解析整個 Big5 固定長度檔案，回傳 DataFrame"""
        records = []
        logger = logging.getLogger(__name__)

        for ln, raw in enumerate(open(infile, 'rb'), start=1):
            if max_lines and ln > max_lines:
                break
            raw = raw.rstrip(b'\r\n')
            if not raw:
                continue
            if len(raw) < self.total_length:
                raw = raw.ljust(self.total_length, b' ')
            rec = {}
            # 基本欄位解析
            for name, start, length, ftype in self.field_defs:
                chunk = raw[start:start+length]
                text = chunk.decode('big5', errors='replace').strip()
                if ftype == 'I':
                    try:
                        rec[name] = int(text)
                    except:
                        rec[name] = None
                else:
                    rec[name] = text or ''

            tx = rec.get('TXTT', '')
            
            # 用於分類判斷的右對齊字串（不儲存到資料庫）
            aa = tx.rjust(210)
            # TT1: 最後 2 字
            rec['TT1'] = aa[-2:]
            # TT2: 最後 4 字
            rec['TT2'] = aa[-4:]

            # 分類函式
            def classify(text, r):
                # 1. tinv (RULB=24)
                if r.get('RULB') == 24:
                    return 1
                # 2. 投資架構
                if any(kw in text for kw in ['投資架構', '組織架構']):
                    return 4
                # 3. 合資
                if '合資' in text:
                    return 2
                # 4. 委建/工程
                if '租地委建' in text or any(kw in text for kw in ['委建', '承攬', '委託興建']) or '工程' in text:
                    return 13
                # 5. 租約
                if all(x not in text for x in ['購買', '出售']) and any(kw in text for kw in ['租', '使用權資產', '租賃資產']):
                    return 19
                # 6. 複合/衍生
                if any(kw in text for kw in ['結構性', '衍生性', '理財產品', '基金', '信託計畫', '受益憑證', '收益憑證',
                                              '理財商品', '資產基礎證券', '組合式商品', '信託單位']):
                    return 8
                # 7. 不動產
                if '合建' in text:
                    return 12
                if '土地銀行' not in text and any(kw in text for kw in ['土地', '建物', '建築物', '基地', '不動產', '地上權', '廠', '用地',
                                                                          '房地', '房產', '房屋', '地號', '小段', '建案', '預售屋', '大樓',
                                                                          '辦公室', '車位', '開發案', '都市更新', '都更', '容移', '購地',
                                                                          '容積', '營運總部']):
                    return 11
                # 8. 設備
                if any(kw in text for kw in ['設備', '固定資產', '生產線', '營業資產', '軟體', '無形資產', '租賃轉讓權', '電站',
                                              '貨櫃', '散裝', '門店', '飛機', '船舶', '其他資產', '冷凍櫃', '新船',
                                              '散貨', '鋪纜', '營業用資產', '售機案', '貨機', '發動機', '客機',
                                              '不良債權', '授信資產', '營運資產', '商標', '專利', '伺服器', '智慧財產']):
                    return 18
                # 9. 併購
                if any(kw in text for kw in ['收購', '合併', '併購', '購併', '分割', '股份轉換']):
                    return 3
                # 10. 領息
                if any(kw in text for kw in ['收益證券', '公司債', '定期存單', '金融債', '債券']):
                    return 7
                # 11. 持股
                if any(kw in text for kw in ['股權', '持股', '普通股', '特別股', '股票', '有價證', '權益', '金融資產',
                                              '認購', '現金增', '增資', '投資', '增發新股', '累計', '設立', '新設',
                                              '籌設', '發行新股', '股份案', '全部股份', '現增']):
                    return 1
                # 12. 公告取得
                if all(kw not in text for kw in ['公告取得資產', '公告取得重大資產']) and any(kw in text for kw in ['公告取得', '取得股份']):
                    return 1
                # 預設 99
                cl0 = 99
                # 次級判別
                t1 = aa[-2:].strip()
                t2 = aa[-4:].strip()
                if '股' in t1:
                    return 2
                if t2 in ['股份', '公司']:
                    return 1
                return cl0

            rec['CL'] = classify(tx, rec)

            records.append(rec)

        if not records:
            logging.getLogger(__name__).warning("沒有讀到任何有效記錄")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # 重新排列欄位
        base = [f[0] for f in self.field_defs]
        extras = ['TT1', 'TT2', 'CL']
        cols = [c for c in base + extras if c in df.columns]
        return df.loc[:, cols]


def main():
    p = argparse.ArgumentParser(description="SBJ_PU11 固定長度檔案解析 (Big5)")
    p.add_argument('-i', '--input',  required=True, help="輸入檔案 (Big5 編碼)")
    p.add_argument('-o', '--output', help="輸出 CSV 檔案 (UTF-8)")
    p.add_argument('--max-lines', type=int, default=None, help="僅解析前 N 行 (預設全部)")
    
    # MySQL 相關參數
    p.add_argument('--to-mysql', action='store_true', help="直接匯入到 MySQL 資料庫")
    p.add_argument('--config', default='config.ini', help="設定檔路徑")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = SBJ_PU11_Parser()
    logging.info(f"開始解析：{args.input}")
    df = parser.parse_file(args.input, args.max_lines)

    if df.empty:
        logging.error("解析結果為空，結束。")
        return

    # 輸出 CSV
    if args.output:
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        logging.info(f"已輸出 {len(df)} 筆到 CSV: {args.output}")

    # 匯入 MySQL
    if args.to_mysql:
        try:
            handler = MySQLHandler(config_file=args.config)
            if handler.connect():
                handler.create_database()
                handler.create_table()
                handler.insert_dataframe(df)
                handler.close()
                logging.info(f"已成功匯入 {len(df)} 筆記錄到 MySQL sbj_pu11 資料表")
            else:
                logging.error("連線 MySQL 失敗。")
        except Exception as e:
            logging.error(f"MySQL 匯入錯誤: {e}")
    
    if not args.output and not args.to_mysql:
        logging.getLogger(__name__).warning("未指定輸出方式，請使用 -o 指定 CSV 檔案或 --to-mysql 匯入資料庫")


if __name__ == '__main__':
    main()
