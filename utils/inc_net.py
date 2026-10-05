import copy
import torch
import logging
from torch import nn
from backbone.linears import TunaLinear

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()

    if '_brmoe' in name:
        from backbone import vit_brmoe
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            d_model=768,
            _device=args["device"][0],

            # BR-MoE settings
            expert_dim=args.get('expert_dim', 16),
            adapter_dim=args.get('adapter_dim', 64),
            topk_routers=args.get('topk_routers', 2),
            topk_experts=args.get('topk_experts', 3),
            init_cls=args["init_cls"],
        )
        if name == "vit_base_patch16_224_brmoe":
            model = vit_brmoe.vit_base_patch16_224_brmoe(num_classes=args["nb_classes"],
                                                         global_pool=False,
                                                         drop_path_rate=args["drop_path"],
                                                         tuning_config=tuning_config)
        elif name == "vit_base_patch16_224_in21k_brmoe":
            model = vit_brmoe.vit_base_patch16_224_in21k_brmoe(num_classes=args["nb_classes"],
                                                               global_pool=False,
                                                               drop_path_rate=args["drop_path"],
                                                               tuning_config=tuning_config)
        else:
            raise NotImplementedError("Unknown type {}".format(name))

        return model

    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class CaRENet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]
       
    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
      
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def generate_fc(self, in_dim, out_dim): 
        fc = TunaLinear(in_dim, out_dim)
        return fc

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        res = self.backbone(x, adapter_id, train, fc_only)
        return res

# ----------------------------------------------------------------------------------------------
# MoSS: frozen pretrained backbone b0 + expert bank / router / shared head (Sec. 3.2)
# ----------------------------------------------------------------------------------------------
def get_frozen_backbone(args):
    """Return (module, out_dim). b0 is frozen for the whole sequence, so stored features stay
    comparable across stages and can be processed by existing and newly added experts."""
    name = args["backbone_type"].lower()
    if name == "identity":
        # Inputs are already feature vectors (e.g. the synthetic subset-sharing benchmark).
        return nn.Identity(), int(args["feature_dim"])
    if name.endswith("_brmoe"):
        raise ValueError("MoSS uses a plain frozen backbone; drop the '_brmoe' suffix "
                         "(e.g. 'vit_base_patch16_224_in21k').")
    import timm
    model = timm.create_model(name, pretrained=args.get("pretrained", True), num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, int(model.num_features)


class MoSSNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        from backbone.moss_moe import ExpertMixture
        self.backbone, self.feature_dim = get_frozen_backbone(args)
        self.moe = ExpertMixture(
            in_dim=self.feature_dim,
            num_classes=args["nb_classes"],
            num_experts=args.get("num_experts_init", 2),
            hidden_dim=args.get("expert_hidden_dim", 256),
            out_dim=args.get("expert_out_dim", 128),
            topk=args.get("topk_experts", 2),
            tau=args.get("router_tau", 1.0),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # b0 is frozen: never enable dropout / stochastic depth in it
        return self

    @torch.no_grad()
    def backbone_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.moe(self.backbone_features(x))
