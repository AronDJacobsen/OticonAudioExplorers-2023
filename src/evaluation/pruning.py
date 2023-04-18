import torch
import torch.nn.utils.prune as prune

def pruning_status(encoder, global_only: bool = False):
    if not global_only:
        print(
            "Sparsity in block1.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block1.conv1.weight == 0))
                / float(encoder.encoder.block1.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block2.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block2.conv1.weight == 0))
                / float(encoder.encoder.block2.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block3.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block3.conv1.weight == 0))
                / float(encoder.encoder.block3.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block4.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block4.conv1.weight == 0))
                / float(encoder.encoder.block4.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in encoder_fc.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder_fc.weight == 0))
                / float(encoder.encoder_fc.weight.nelement())
            )
        )
        print(
            "Sparsity in latent_classifier.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.latent_classifier.weight == 0))
                / float(encoder.latent_classifier.weight.nelement())
            )
        )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(encoder.encoder.block1.conv1.weight == 0)
                + torch.sum(encoder.encoder.block2.conv1.weight == 0)
                + torch.sum(encoder.encoder.block3.conv1.weight == 0)
                + torch.sum(encoder.encoder.block4.conv1.weight == 0)
                + torch.sum(encoder.encoder_fc.weight == 0)
                + torch.sum(encoder.latent_classifier.weight == 0)
            )
            / float(
                encoder.encoder.block1.conv1.weight.nelement()
                + encoder.encoder.block2.conv1.weight.nelement()
                + encoder.encoder.block3.conv1.weight.nelement()
                + encoder.encoder.block4.conv1.weight.nelement()
                + encoder.encoder_fc.weight.nelement()
                + encoder.latent_classifier.weight.nelement()
            )
        )
    )


def prune_encoder_(encoder_, pruning_ratio):
    # Determine pruning parameters
    parameters_to_prune = (
        (encoder_.encoder.block1.conv1, 'weight'),
        (encoder_.encoder.block2.conv1, 'weight'),
        (encoder_.encoder.block3.conv1, 'weight'),
        (encoder_.encoder.block4.conv1, 'weight'),
        (encoder_.encoder_fc, 'weight'),
        (encoder_.latent_classifier, 'weight'),
    )

    # Run pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    pruning_status(encoder_, global_only=True)