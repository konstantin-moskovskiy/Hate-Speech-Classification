python ../src/data/make_dataset.py -o "../data/external/Ethos_Dataset_Binary.csv"
python ../src/features/build_features.py -i "../data/external/Ethos_Dataset_Binary.csv" -o "../data/interim/Ethos_Dataset_Binary_pr.csv"
python ../src/models/train_predict_model.py -i "../data/interim/Ethos_Dataset_Binary_pr.csv" -r "../models/results.csv" -m "../models/final_model.pkl"