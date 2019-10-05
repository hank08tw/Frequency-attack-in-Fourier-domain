from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=224,height=224):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

all_names=os.listdir('./check_alexnet_root')
for all_name in all_names:
    if all_name== '.DS_Store':continue
    convertjpg('./check_alexnet_root/'+all_name,'./check_alexnet_root_resize')