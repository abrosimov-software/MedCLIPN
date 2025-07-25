import torch
import torch.nn.functional as F

class CLIPNLoss:
    def __init__(self, temperature=0.07, mode="L2"):
        """
        CLIPN Loss implementation with Image-Text Binary-Opposite Loss (ITBO) 
        and Text Semantic-Opposite Loss (TSO).
        
        Args:
            temperature (float): Temperature parameter for similarity scaling
        """
        self.temperature = temperature
        self.mode = mode
    
    def compute_itbo_loss(self, image_features, text_features, text_features_no):
        """
        Computes the Image-Text Binary-Opposite Loss (ITBO).

        Args:
            image_features (torch.Tensor): Normalized image features of shape [N, D].
            text_features (torch.Tensor): Normalized standard text features of shape [N, D].
            text_features_no (torch.Tensor): Normalized "no" text features of shape [N, D].

        Returns:
            torch.Tensor: The scalar ITBO loss.
        """
        N = image_features.shape[0]
        
        # Compute similarities
        sim_img_text = torch.matmul(image_features, text_features.T) / self.temperature
        sim_img_text_no = torch.matmul(image_features, text_features_no.T) / self.temperature
        
        # Stack similarities for softmax calculation
        logits_combined = torch.stack([sim_img_text, sim_img_text_no], dim=-1) # Shape: [N, N, 2]
        probs = F.softmax(logits_combined, dim=-1) # Shape: [N, N, 2]

        # p_yes = probs[:, :, 0]
        p_no = probs[:, :, 1]

        eyes = torch.eye(N, device=image_features.device, dtype=torch.bool)

        loss_bin_no = (-torch.log(p_no[~eyes] + 1e-8)).view(-1).sum() / (N**2 - N)
        loss_bin_yes_oppose = (-torch.log(1 - p_no[eyes] + 1e-8)).view(-1).sum() / N

        return loss_bin_no + loss_bin_yes_oppose
        

    def compute_tso_loss(self, text_features, text_features_no):
        """
        Computes the Text Semantic-Opposite Loss (TSO).

        Args:
            text_features (torch.Tensor): Normalized standard text features of shape [N, D].
            text_features_no (torch.Tensor): Normalized "no" text features of shape [N, D].

        Returns:
            torch.Tensor: The scalar TSO loss.
        """
        if self.mode == "L2":
            l2_distances = 2 - 2 * (text_features * text_features_no).sum(-1) + 1e-8
            tso_loss = 2 - torch.sqrt(l2_distances)
        elif self.mode == "cosine":
            tso_loss = (text_features * text_features_no).sum(-1) + 1
        
        return tso_loss.mean()

    def __call__(self, image_features, text_features, text_features_no):
        """
        Computes the total CLIPN loss.

        Args:
            image_features (torch.Tensor): Image features of shape [N, D].
            text_features (torch.Tensor): Standard text features of shape [N, D].
            text_features_no (torch.Tensor): "No" text features of shape [N, D].

        Returns:
            tuple: A tuple containing the total loss (torch.Tensor) and a dictionary of loss components.
        """
        # Ensure features are L2-normalized
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        text_features_no = F.normalize(text_features_no, p=2, dim=1)

        # Compute individual losses
        itbo_loss = self.compute_itbo_loss(image_features, text_features, text_features_no)
        tso_loss = self.compute_tso_loss(text_features, text_features_no)
        
        # Total loss is the sum of both components
        total_loss = itbo_loss + tso_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'itbo_loss': itbo_loss.item(),
            'tso_loss': tso_loss.item()
        }
        
        return total_loss, loss_dict

