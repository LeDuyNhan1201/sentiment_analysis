# sentiment_analysis
install python 3.9.xx
pip install -r requirements.txt

# Tien xu ly file data/sentiment_analysis.csv va luu vao file sentiment_data.csv
python pre_process.py

# Chia data thanh 2 phan train va test & luu vocab vao vocab.json
python data.py

# Xay dung model va luu vao folder model/
python train_eval.py

# Test xem model vua huan luyen co chinh xac khong
python test.py
