import matplotlib.pyplot as plt
import numpy as np

models = [
    'LSTM','GRU','GCN-LSTM','Transformer',
    'MultiStep-LSTM','MultiStep-Transformer',
    'STGCN','AGC-STGCN','MultiScale-STGCN',
    'Adaptive-STGCN','AdaptiveMultiScale-STGCN',
    'PI-Adaptive-STGCN','CAH-STGCN','LHSA-STGCN'
]

metr = [0.0933176,0.0939955,0.0974625,0.1187215,
        0.1481815,0.2102233,0.0688380,0.0669291,
        0.0715101,0.0435512,0.0423141,0.0446237,
        0.0392240,0.0409552]

bay = [0.0279295,0.0288322,0.0345572,0.0242514,
       0.0281251,0.0288243,0.0456680,0.0128584,
       0.0131055,0.0139829,0.0136745,0.0134127,
       0.0859311,0.0126277]

pems = [0.0391553,0.0407845,0.0400853,0.0370150,
        0.0393330,0.0415942,0.0397751,0.0187833,
        0.0180091,0.0189835,0.0208000,0.0187571,
        0.0179609,0.0179098]

x = np.arange(len(models))
w = 0.25

plt.figure(figsize=(14,6))
plt.bar(x-w, metr, w, label='METR-LA')
plt.bar(x, bay, w, label='PEMS-BAY')
plt.bar(x+w, pems, w, label='PEMS04')

plt.xticks(x, models, rotation=60, ha='right')
plt.ylabel('MAE')
plt.title('Cross-Dataset MAE Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('mae_comparison.png', dpi=300)
plt.show()