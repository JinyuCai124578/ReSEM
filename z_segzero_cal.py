import json
from utils.dataset import ValDataset_EM
from utils.utils import evaluate_text_metrics, AverageMeter, Summary
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

def get_class_name(one_eval_result, val_data_list):
    """
    根据eval结果中的prompt匹配val_data_list中的question，返回对应的class_name
    
    参数:
        one_eval_result: 单个eval结果数据点(dict)
        val_data_list: val数据集列表(list of dict)
        
    返回:
        匹配到的class_name(str)，如果没有匹配则返回None
    """
    # 获取eval结果中的prompt
    target_prompt = one_eval_result.get("prompt")
    if not target_prompt:
        return None
    
    # 遍历val_data_list
    for data in val_data_list:
        # 检查image_name是否匹配
        if data.get("image_name") == one_eval_result.get("image_name"):
            # 遍历shapes中的qa_list
            for shape in data.get("shapes", []):
                for qa in shape.get("qa_list", []):
                    if qa.get("question") == target_prompt:
                        return shape.get("class_name"), qa.get("answer")
    return None, None


def get_mask_path(one_eval_result, val_data_list):
    """
    根据eval结果中的prompt匹配val_data_list中的question，返回对应的class_name
    
    参数:
        one_eval_result: 单个eval结果数据点(dict)
        val_data_list: val数据集列表(list of dict)
        
    返回:
        匹配到的class_name(str)，如果没有匹配则返回None
    """
    # 获取eval结果中的prompt
    target_prompt = one_eval_result.get("prompt")
    if not target_prompt:
        return None
    
    # 遍历val_data_list
    for data in val_data_list:
        # 检查image_name是否匹配
        if data.get("image_name") in one_eval_result.get("image"):
            # 遍历shapes中的qa_list
            for shape in data.get("shapes", []):
                for qa in shape.get("qa_list", []):
                    if qa.get("question") in target_prompt or target_prompt in qa.get("question"):
                        mask_path=one_eval_result.get("image").replace(data.get("image_name"), shape.get("mask_name"))
                        return mask_path, shape.get("color_id", None)
    return None, None

def get_image_path(one_eval_result, val_data_list):
    mask_path=one_eval_result.get("mask")
    for data in val_data_list:
        shapes=data.get("shapes", [])
        for shape in shapes:
            if shape.get("mask_name") in mask_path:
                image_name= data.get("image_name")
                image_path=mask_path.replace(shape.get("mask_name"), image_name)
                return image_path




f1_meter = AverageMeter("F1", ":6.3f", Summary.SUM)
bleu_meter=AverageMeter("Bleu", ":6.3f", Summary.SUM)
cider_meter=AverageMeter("CIDEr", ":6.3f", Summary.SUM)
bertscorep_meter=AverageMeter("BERTScore_P", ":6.3f", Summary.SUM)
bertscorer_meter=AverageMeter("BERTScore_R", ":6.3f", Summary.SUM)
bertscoref1_meter=AverageMeter("BERTScore_F1", ":6.3f", Summary.SUM)


val_data_list = ValDataset_EM(
                '/mnt/shared-storage-user/caijinyu/data',
                None,
                "/mnt/shared-storage-user/caijinyu/model/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
                "organelle||plantorgan||cremi||material"+"_val",
                1024,
                use_gpt_qa=True,
            ).json_data_list

test_data_list = ValDataset_EM(
                '/mnt/shared-storage-user/caijinyu/data',
                None,
                "/mnt/shared-storage-user/caijinyu/model/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
                "organelle||plantorgan||cremi||material"+"_test",
                1024,
                use_gpt_qa=True,
            ).json_data_list

val_data_list+=test_data_list

## find segzero
segzero_mask_folder="/mnt/shared-storage-user/ai4sdata2-share/caijinyu/eval_results/segzero_em/predicted_masks"
eval_result_list=json.load(open("/mnt/shared-storage-user/ai4sdata2-share/caijinyu/eval_results/segzero_em/eval_sample_results.json"))
target_result_list=json.load(open('/home/caijinyu/LISA/chat_sample_better.json'))
for one_target_result in tqdm(target_result_list[-2:]):
    for one_eval_result in eval_result_list:
        if one_eval_result["image_name"] in one_target_result["image"]:
            pred_mask_path=one_eval_result['pred_mask_path']
            mask_name=pred_mask_path.split('/')[-1].split('.')[0]
            mask_np=np.array(Image.open(pred_mask_path))!=0
            # import pdb;pdb.set_trace()
            image_path=one_eval_result['image_path']
            if "tiff" in image_path:
                image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
                image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
                image_np=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
            else:
                image_np = cv2.imread(image_path)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            save_img=image_np.copy()
            save_img[mask_np] = (
                image_np * 0.5
                + mask_np[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[mask_np]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            image_name=image_path.split('.')[0].split('/')[-1]
            save_path="/home/caijinyu/LISA/vis_output/segzero/{}.jpg".format(mask_name)
            cv2.imwrite(save_path, save_img)
            image_save_path="/home/caijinyu/LISA/vis_output/segzero/{}.jpg".format(image_name)
            cv2.imwrite(image_save_path, image_np)
### get image
# chat_sample_list=json.load(open("/home/caijinyu/LISA/chat_sample_better.json"))
# for one_chat_sample in tqdm(chat_sample_list):
#     image_path=get_image_path(one_chat_sample, val_data_list)
#     one_chat_sample.update({"image": image_path})

# with open("/home/caijinyu/LISA/chat_sample_better.json", "w") as f:
#     json.dump(chat_sample_list, f, indent=4)
# print(json.dumps(chat_sample_list, indent=4))

### segzero
# eval_result="/mnt/shared-storage-user/ai4sdata2-share/caijinyu/eval_results/segzero_em/eval_sample_results.json"
# eval_result_list=json.load(open(eval_result))

# for one_eval_result in tqdm(eval_result_list):
#     class_name, text_output_gt =get_class_name(one_eval_result, val_data_list)
#     assert class_name is not None
#     f1=1 if class_name in one_eval_result['raw_model_output'] else 0
#     text_metrics=evaluate_text_metrics(candidate=one_eval_result['thinking_process'], reference=text_output_gt)
#     # 将结果添加到原始数据中
#     one_eval_result.update({
#         "class_name": class_name,
#         "f1_score": f1,
#         "text_metrics": text_metrics
#     })
#     f1_meter.update(f1)
#     bleu_meter.update(text_metrics['BLEU'])
#     cider_meter.update(text_metrics['CIDEr'])
#     bertscorep_meter.update(text_metrics['BERTScore_P'])
#     bertscorer_meter.update(text_metrics['BERTScore_R'])
#     bertscoref1_meter.update(text_metrics['BERTScore_F1'])

# bleu_meter.all_reduce()
# cider_meter.all_reduce()
# bertscorep_meter.all_reduce()
# bertscorer_meter.all_reduce()
# bertscoref1_meter.all_reduce()
# f1_meter.all_reduce()

# bleu= bleu_meter.avg
# cider = cider_meter.avg
# bertscorep = bertscorep_meter.avg
# bertscorer = bertscorer_meter.avg
# bertscoref1 = bertscoref1_meter.avg
# f1= f1_meter.avg

# print({'bleu': bleu, 'cider': cider, 'bert_score_p': bertscorep, 'bert_score_r': bertscorer, 'bert_score_f1': bertscoref1, 'f1': f1})
# output_path="/home/caijinyu/LISA/vis_output/segzero/eval_result.json"
# with open(output_path, 'w') as f:
#     json.dump(eval_result_list, f, indent=4)


### overlap gt generate

chat_sample_list=json.load(open("/home/caijinyu/LISA/chat_sample_better.json"))
for chat_sample in chat_sample_list[-2:]:
    mask_path,color_id=get_mask_path(chat_sample, val_data_list)
    if mask_path is None:
        print(("mask_path is None, check", chat_sample))
        continue
    chat_sample.update({"mask": mask_path, "color_id": color_id})
    if "ceramic" in mask_path.lower() or "nanoparticle" in mask_path.lower():
        mask_np=np.array(Image.open(mask_path).convert('L'))==int(color_id) if color_id is not None \
                else np.array(Image.open(mask_path).convert('L'))!=0
    else:
        mask_np=np.array(Image.open(mask_path))==int(color_id) if color_id is not None \
                else np.array(Image.open(mask_path))!=0
    # import pdb;pdb.set_trace()
    image_path=chat_sample.get("image")
    if "tiff" in image_path:
        image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
        image_np=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    else:
        image_np = cv2.imread(image_path)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    save_img=image_np.copy()
    save_img[mask_np] = (
        image_np * 0.5
        + mask_np[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
    )[mask_np]
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    image_name=image_path.split('.')[0].split('/')[-1]
    class_name=chat_sample.get("class")
    save_path="/home/caijinyu/LISA/vis_output/gt/{}_{}_masked_img.jpg".format(image_name,class_name)
    cv2.imwrite(save_path, save_img)
    save_img_path="/home/caijinyu/LISA/vis_output/gt/{}_{}.jpg".format(image_name,class_name)
    cv2.imwrite(save_img_path, image_np)

    

# output_path="/home/caijinyu/LISA/chat_sample_good_with_mask.json"
# with open(output_path, 'w') as f:
#     json.dump(chat_sample_list, f, indent=4)


