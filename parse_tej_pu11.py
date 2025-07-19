#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P_U11 固定長度 FOC 檔案解析器
依照說明檔，以 bytes 精準切割、Big5 解碼，並可輸出 CSV 或匯入 MySQL。
"""

import argparse
import logging
from datetime import datetime

import pandas as pd
from create_mysql_db import MySQLHandler  # 可重用既有 MySQLHandler

class PU11Parser:
    def __init__(self):
        # (欄位名稱, 起始 byte, 長度, 類型 'A'=文字 'I'=數字)
        self.field_defs = [
            ('BAN',      0,   8,  'A'),
            ('CODE',     8,   7,  'A'),
            ('NAME',    15,  20,  'A'),
            ('GDATE',   35,   8,  'I'),
            ('HHMMSS',  43,   6,  'I'),
            ('DATE',    49,   8,  'I'),
            ('OD',      57,   2,  'I'),
            ('HR_REALS',59,   6,  'I'),
            ('FILE_NM', 65,  70,  'A'),
            ('OCCUR_D',135,   8,  'I'),
            ('SPOKER', 143,  12,  'A'),
            ('D_REALS',155,   8,  'I'),
            ('KEYIN1', 163,   8,  'I'),
            ('KEY_HR',  171,   4,  'I'),
            ('RULA',   175,   3,  'I'),
            ('RULB',   178,   3,  'I'),
            ('DBCL',   181,   9,  'A'),
            ('MKT',    190,   3,  'A'),
            ('NO',     193,   5,  'A'),
            ('TXT',    198,  70,  'A'),
        ]
        # 記錄長度
        self.record_len = max(start + length for _, start, length, _ in self.field_defs)

    def parse_file(self, infile: str, max_lines: int = None) -> pd.DataFrame:
        records = []
        logger = logging.getLogger(__name__)

        with open(infile, 'rb') as fp:
            for ln, raw in enumerate(fp, start=1):
                if max_lines and ln > max_lines:
                    break
                raw = raw.rstrip(b'\r\n')
                if not raw:
                    continue
                # 若不夠長，補空白
                if len(raw) < self.record_len:
                    raw = raw.ljust(self.record_len, b' ')
                rec = {}
                # 切欄位
                for name, start, length, ftype in self.field_defs:
                    chunk = raw[start:start+length]
                    text = chunk.decode('big5', errors='replace').strip()
                    if ftype == 'I':
                        try:
                            rec[name] = int(text)
                        except ValueError:
                            rec[name] = None
                    else:
                        rec[name] = text
                # 衍生欄位
                hr = rec.get('HR_REALS')
                rec['HM_ANN'] = hr // 100 if isinstance(hr, int) else None
                dbcl = rec.get('DBCL', '')
                rec['CLA']    = dbcl[:1] if dbcl else ''
                records.append(rec)

        if not records:
            logger.error("未讀到任何記錄，請確認檔案格式。")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # 重新排序欄位：原欄位 + 衍生
        cols = [f[0] for f in self.field_defs] + ['HM_ANN', 'CLA']
        return df.loc[:, [c for c in cols if c in df.columns]]

def main():
    p = argparse.ArgumentParser(description="P_U11 FOC 固定長度檔解析")
    p.add_argument('-i','--input',    required=True, help="輸入 FOC 檔案")
    p.add_argument('-o','--output',   help="輸出 CSV 檔案")
    p.add_argument('--max-lines', type=int, default=None, help="僅解析前 N 筆")
    p.add_argument('--to-mysql', action='store_true', help="直接匯入 MySQL")
    p.add_argument('--config', default='config.ini', help="設定檔路徑")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = PU11Parser()
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
                handler.create_tej_pu11_table()
                handler.insert_tej_pu11_dataframe(df)
                handler.close()
                logging.info(f"已成功匯入 {len(df)} 筆記錄到 MySQL tej_pu11_1 資料表")
            else:
                logging.error("連線 MySQL 失敗。")
        except Exception as e:
            logging.error(f"MySQL 匯入錯誤: {e}")

if __name__ == '__main__':
    main()
