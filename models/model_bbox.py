from models import XVLMBase, load_pretrained
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_relative_position_bias_table(state_dict, model):
    """
    Interpolates the relative position bias table to match the new model shape.
    """
    for name, param in model.state_dict().items():
        if "relative_position_bias_table" in name and name in state_dict:
            old_bias = state_dict[name]  # Load from checkpoint
            new_bias = param  # Current model's expected shape

            if old_bias.shape != new_bias.shape:
                print(f"Interpolating {name} from {old_bias.shape} to {new_bias.shape}")

                num_heads = old_bias.shape[1]  # Get number of attention heads

                # Compute original and target sizes
                src_size = old_bias.shape[0]  # Original number of relative positions (e.g., 23)
                dst_size = new_bias.shape[0]  # Target number of relative positions (e.g., 529)

                # Create a mapping from old to new relative positions
                # Assuming the relative positions are linear indices, we can interpolate directly
                x_old = np.linspace(0, 1, src_size)
                x_new = np.linspace(0, 1, dst_size)

                new_values = []

                for i in range(num_heads):
                    # Interpolate for each attention head
                    interpolator = RegularGridInterpolator((x_old,), old_bias[:, i].numpy(), method="linear")
                    new_bias_1d = interpolator((x_new,))
                    new_values.append(torch.tensor(new_bias_1d).contiguous().view(-1, 1))

                # Replace with interpolated values
                state_dict[name] = torch.cat(new_values, dim=-1).to(param.device)

    return state_dict

class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=True)
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, load_bbox_pretrain=False, is_eval=False):
        print("### load_bbox_pretrain, ", load_bbox_pretrain)
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        state_dict = interpolate_relative_position_bias_table(state_dict, self)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, target_bbox=None):
        image_embeds, _ = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        output_coord = self.predict_bbox(image_embeds, text_embeds, text_atts)
        # output_coord & target_bbox: 64, 4

        if target_bbox is None:
            return output_coord

        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox)

        return output_coord, loss_bbox, loss_giou

