# DVC
git init
dvc init  ( we use dvc+supporting SCM tool always here)


<!-- add dvc :: letting dvc to manage data_folder-->
dvc add path/data_folder/



```sh
dvc config core.analytics false
dvc config core.autostage true  # autotage
```

<!-- space check -->
dvc du path/data_folder/

<!-- check dvc remote -->
dvc remote list


<!-- GOOGLE DRive -->
```sh
$ Google API Console
>> Enable Drive API
>> create credentials >> user-data >> APP-NAme, Developer-contact, Test-User SAVE
>> **CLIENT-ID** and **CLIENT-SECRET**
>> **API KEY**
```

<!-- add gdrive -->
dvc remote add --default remote-name gdrive:/GDRIVE-ID
dvc remote modify gdrive gdrive_acknowledge_abuse true

dvc remote modify gdrive gdrive_clien_id CLIENT-ID
dvc remote modify gdrive gdrive_clien_secret  CLIENT-SECRET

<!-- push to dgrive and manage -->
dvc push -r gdrive


 whenever data change and commit dvc md5 hash changes, do  git commit `best practise` 

```sh
dvc commit
git commit
```

<-- pull it from dvc -->
dvc pull -r local

<-- checkout -->
git checkout ...
dvc checkout


<!-- Data Pipelines:: dvc stage -->
dvc stage 
    -n `name of the stage to add`  '--name name'
    -p `Declare parameter to use as additional dependency`  '-params [<filename>:]<params_list>'
    -d `Declare dependencies for reproducible cmd`  '--deps <path>'
    -o `Declare output file or directory`  '--outs <filename>'
    src/python.py


```sh
dvc stage add 
    -n train 
    -d src/train.py 
    -d configs/experiment/catdog.yaml 
    -o logs 
    -o outputs 
python src/train.py data.batch_size=64 model.pretrained=false trainer.max_epochs=10 logger=comet
```
edit:: `data.dvc`

<!-- reproduce -->
dvc repro




**[pull from container](https://stackoverflow.com/questions/76306644/how-can-i-reauthorize-dvc-with-google-drive-for-remote-storage)**
**err: [configuration](https://discuss.dvc.org/t/error-configuration-error-gdrive-remote-auth-failed-with-credentials-in-gdrive-credentials-data/1254)**


----



# [[S3]] + [[dvc]]
```sh
pip install dvc[s3]
```

> [!IMPORTANT] create IAM user
> - AdministratorAccess
> - AmazonS3FullAccess


`note:` setup [[AWS CLI]] in the machine
```sh
aws configure
> AWS ACCESS_KEY_ID
> AWS SECRET_ACCESS_KEY
> DEFAULT REGION NAME: ap-south-1
> DEFAULT OUTPUT_FORMAT: json
```

```sh
aws s3 ls
```

```sh
git init 
dvc init
git add README.md
git commit -m "init commit"
git branch -M main

# DVC
dvc add path_to_dataset/
dvc remote add -d myremote s3://bucket-name 
dvc push
dvc pull
```