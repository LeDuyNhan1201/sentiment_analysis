import csv

# def extract_text_and_label(input_file, output_file):
#     with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
#             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
#
#         next(reader)  # Skip the header row in the input file
#
#         # Write new header to output
#         writer.writerow(['text', 'label'])
#
#         for row in reader:
#             if len(row) >= 2:  # Ensure at least 2 columns
#                 label = row[-2][0].upper() + row[-2][1:]
#                 text = row[-3]
#                 writer.writerow([text, label])
#
#
# extract_text_and_label(
#     'data/sentiment_analysis.csv',
#     'sentiment_data.csv'
# )

import pandas as pd

def remove_duplicates_from_csv(input_path, output_path=None, subset=None):
    """
    Xoá các dòng bị trùng trong file CSV.

    Args:
        input_path (str): Đường dẫn đến file CSV gốc.
        output_path (str): Đường dẫn để lưu file mới. Nếu None thì ghi đè lên file gốc.
        subset (list or str): Danh sách các cột dùng để so sánh trùng lặp. Nếu None, so sánh toàn bộ dòng.

    Returns:
        int: Số dòng sau khi xoá trùng.
    """
    df = pd.read_csv(input_path)
    before = len(df)

    df = df.drop_duplicates(subset=subset)

    after = len(df)
    print(f"[INFO] Đã xoá {before - after} dòng trùng. Còn lại {after} dòng.")

    output = output_path or input_path
    df.to_csv(output, index=False)
    return after



remove_duplicates_from_csv("sentiment_data.csv")
