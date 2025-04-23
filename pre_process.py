import csv

def extract_text_and_label(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
            open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        writer.writerow(['text', 'label'])

        for row in reader:
            if len(row) >= 2:  # Ensure at least 2 columns
                label = row[-2]
                text = row[-1]
                writer.writerow([text, label])


extract_text_and_label(
    'data/twitter_training.csv',
    'sentiment_data.csv'
)

