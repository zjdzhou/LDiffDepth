import torch
import datasets
import networks

def get_monovit_pretrained(load_type):
    depth = networks.DeepNet('mpvitnet')
    # 加载方式也不同，在1024分年率的时候Monovit的作者只提供了一个权重文件
    if load_type == 1024:
        depth_dict = torch.load(r"D:\rd_F\mobilevit_distilla_pose_nofreeze_light_decoder\models\monovit_1024\depth.pth")
        # new_dict = depth_dict
        # 调整预训练模型的权重字典
        new_dict = {}
        for k, v in depth_dict.items():
            name = k[7:]
            new_dict[name] = v
        depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})

    if load_type == 640:
        # 加载编码器的权重
        mpvit_encoder_dict = torch.load(r"D:\Root\rd_F\mobilevit_distilla_pose_nofreeze_light_decoder\models\monovit_640\encoder.pth")
        model_dict = depth.encoder.state_dict()
        depth.encoder.load_state_dict({k: v for k, v in mpvit_encoder_dict.items() if k in model_dict})
        # 加载解码器的权重
        mpvit_decoder_path = r"D:\Root\rd_F\mobilevit_distilla_pose_nofreeze_light_decoder\models\monovit_640\depth.pth"
        depth.decoder.load_state_dict(torch.load(mpvit_decoder_path))


    return depth
