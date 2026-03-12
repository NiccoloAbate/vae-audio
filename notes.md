

## Commands

* python train.py -c config_raw_esc50.json

* Small
    * python demo.py -r saved/models/SpecVAE/0307_134517/model_best.pth
    * python demo.py -r saved/models/SpecVAE/0305_115842/model_best.pth
    * python demo_raw.py -r saved/models/RawAudioVAE/0309_220652/checkpoint-epoch150.pth
    * python demo_raw.py -r saved/models/RawAudioVAE/0309_233328/checkpoint-epoch680.pth
* ESC-50
* Medley
    * python demo_raw.py -r saved/models/RawAudioVAE/0311_013216/model_best.pth
    * python demo_raw.py -r saved/models/RawAudioVAE/0311_150156/model_best.pth

* python evaluate_raw.py -r saved/models/RawAudioVAE/0309_233328/checkpoint-epoch680.pth

* python plot_training.pyc



## Thoughts
* Descriminator (adversarial training) is basically a normal conv1d classifier (I have done one of those!) which learns at the same time as the vae
* "Feature matching" means taking the weights in the descriminator model and using those as features to evaluate the perceptual quality of the vae