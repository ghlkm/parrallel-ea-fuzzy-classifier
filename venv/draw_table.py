import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
clust_data=(
(' avila-tr_150g_100Npop_10f_1n_copy ', 0.2239 ),
(' avila-tr_150g_100Npop_10f_3n_copy ', 0.3102 ),
(' avila-tr_150g_100Npop_5f_1n_copy ', 0.2186 ),
(' avila-tr_150g_100Npop_5f_3n_copy ', 0.2901 ),
(' Breast_w_150g_100Npop_10f_1n_copy ', 0.0153 ),
(' Breast_w_150g_100Npop_10f_3n_copy ', 0.0254 ),
(' Breast_w_150g_100Npop_5f_1n_copy ', 0.0146 ),
(' Breast_w_150g_100Npop_5f_3n_copy ', 0.0274 ),
(' gesture_150g_100Npop_10f_1n_copy ', 0.0183 ),
(' gesture_150g_100Npop_10f_3n_copy ', 0.0250 ),
(' gesture_150g_100Npop_5f_1n_copy ', 0.0163 ),
(' gesture_150g_100Npop_5f_3n_copy ', 0.0283 ),
(' Glass_150g_100Npop_10f_1n_copy ', 0.0044 ),
(' Glass_150g_100Npop_10f_3n_copy ', 0.0112 ),
(' Glass_150g_100Npop_5f_1n_copy ', 0.0050 ),
(' Glass_150g_100Npop_5f_3n_copy ', 0.0110 ),
(' sat_150g_100Npop_10f_1n_copy ', 0.1208 ),
(' sat_150g_100Npop_10f_3n_copy ', 0.1790 ),
(' sat_150g_100Npop_5f_1n_copy ', 0.1131 ),
(' sat_150g_100Npop_5f_3n_copy ', 0.1466 ),
)
colLabels=("program setting", "avg migration time spent")
ax.table(cellText=clust_data,
          colLabels=colLabels,
          loc='center',
         rowLoc='right',
         cellLoc='right',)
fig.tight_layout()
plt.show()