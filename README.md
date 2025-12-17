# LN-UXFormer: Local and Non-Local U-Net Cross Transformer Fusion Network

<img width="1240" height="598" alt="image" src="https://github.com/user-attachments/assets/749646b8-0c05-41db-8c1b-77d64da2186b" />

## 📋 Overview

A bidirectional interactive learning framework that establishes explicit parallel pathways for mutual information exchange between local feature extraction (U-Net) and non-local contextual modeling (Transformer) for medical image segmentation.

## 🚀 Key Features

- ✅ Bidirectional CNN-Transformer fusion
- ✅ Cross-Attention (XATTN) for local-to-non-local transfer
- ✅ Cross CBAM (XCBAM) for non-local-to-local transfer
- ✅ Multi-scale feature integration
- ✅ State-of-the-art performance on 5 medical datasets

## 📊 Results

<img width="1183" height="676" alt="image" src="https://github.com/user-attachments/assets/917bcef1-3080-4839-98ef-728acd817521" />

<img width="767" height="355" alt="image" src="https://github.com/user-attachments/assets/3f2db718-03ea-4bb3-8f00-e20ee10b15f0" />

<img width="1183" height="298" alt="image" src="https://github.com/user-attachments/assets/04c8f70f-b955-4ebe-b8f9-fb1b34f1d603" />

<img width="759" height="258" alt="image" src="https://github.com/user-attachments/assets/c8b94113-7222-46f3-b986-1984771ffa3c" />

## 🎯 Quick Start

### Training

Open and run `train.ipynb`:

from data_load import Dataset

from model.LN_UXFormer import LN_UXFormer

model = LN_UXFormer(in_channels=1, num_classes=1)


### Testing

Open and run `test.ipynb`:

# Load checkpoint and evaluate

checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint)

### File
LN_UXFormer/
├── model/
│   ├── LN_UXFormer.py           # Main model architecture
│   ├── swin_transformer.py      # Swin Transformer implementation
│   └── transformer/             # Transformer components
│
├── data/
│   ├── Training/                # Training dataset
│   └── Testing/                 # Testing dataset
│
├── result_bestmodel/
│   └── LN_UXFormer_epoch.pt    # Saved model checkpoints
│
├── result_image/                # Output visualization results
│
├── data_load.py                 # Dataset loader
├── train.ipynb                  # Training notebook
├── test.ipynb                   # Testing notebook
├── requirements.txt             # Package dependencies
└── README.md                    # Project documentation

## 🙏 Acknowledgments

This work was supported by:
- National Research Foundation of Korea (NRF) grant (RS-2024-00357917)
- IITP - Innovative Human Resource Development for Local Intellectualization (IITP-2025-RS-2022-00156287)
- IITP - ICAN program (IITP-2025-RS-2022-00156385)

All grants are funded by the Korean government (MSIT).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
