import os, shutil
from sklearn.model_selection import train_test_split

val_size = 0.2
postfix = 'jpg'
imgpath = r'd:\study\rdk_docker\dataset\trainval\VOC2007\JPEGImages'
txtpath =  r'd:\study\rdk_docker\dataset\trainval\VOC2007\Yolo_labels'

output_train_img_folder =r'd:\study\rdk_docker\dataset\trainval\VOC2007\train\images'
output_val_img_folder =  r'd:\study\rdk_docker\dataset\trainval\VOC2007\val\images'
output_train_txt_folder =  r'd:\study\rdk_docker\dataset\trainval\VOC2007\train\labels'
output_val_txt_folder =  r'd:\study\rdk_docker\dataset\trainval\VOC2007\val\labels'

os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)


listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, val = train_test_split(listdir, test_size=val_size, shuffle=True, random_state=0)

#todo：需要test放开

# train, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=0)
# train, val = train_test_split(train, test_size=val_size, shuffle=True, random_state=0)

for i in train:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_train_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_train_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)

for i in val:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_val_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_val_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)