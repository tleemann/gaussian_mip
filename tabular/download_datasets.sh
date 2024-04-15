data_folder=./datasets
mkdir ${data_folder}/purchase
wget -O ${data_folder}/purchase/purchase.tgz https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz 
tar zxvf ${data_folder}/purchase/purchase.tgz -C ${data_folder}/purchase
mv ${data_folder}/purchase/dataset_purchase ${data_folder}/purchase/purchase.csv
rm ${data_folder}/purchase/purchase.tgz