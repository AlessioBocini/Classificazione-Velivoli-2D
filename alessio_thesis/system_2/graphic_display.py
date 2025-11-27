import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact

def display_graph(tests, LOS=True):
    classes = list(range(21))
    test_names = list(tests.keys())
    mat = np.array([tests[k] for k in test_names]).T  # classi × test
    macro = [float(np.mean(tests[k])) * 100 for k in test_names]

    def show_graph(view):
        plt.figure(figsize=(9,6))
        if view == "Macro-average":
            plt.plot(range(1, len(macro)+1), macro, marker="o")
            plt.title(f"Macro-average Precision (%) — LOS={LOS}")
            plt.xlabel("Test #"); plt.ylabel("Precision (%)")
            plt.xticks(range(1, len(test_names)+1), test_names)
            plt.ylim(0, 100); plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

        elif view == "Trend per Classe":
            plt.figure(figsize=(10,6))
            for i, c in enumerate(classes):
                plt.plot(range(1, len(test_names)+1),
                        np.array(mat[i])*100,
                        linewidth=1,
                        label=f"Modello {c}") 

                plt.title(f"Precision Trend per Classe — LOS={LOS}")
                plt.xlabel("Test #"); plt.ylabel("Precision (%)")
                plt.xticks(range(1, len(test_names)+1), test_names)
                plt.ylim(0, 100)
                plt.grid(True, linestyle="--", alpha=0.4)

                # 🔹 Legenda esterna
                plt.legend(title="Modelli", bbox_to_anchor=(1.02, 1),
                        loc="upper left", fontsize="small", ncol=2)

                plt.tight_layout()

        elif view == "Heatmap":
            plt.imshow(mat, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
            plt.title(f"Precision Heatmap — Classi × Test — LOS={LOS}")
            plt.xlabel("Test"); plt.ylabel("Class")
            plt.xticks(ticks=range(len(test_names)), labels=test_names)
            plt.yticks(ticks=range(len(classes)), labels=classes)
            cbar = plt.colorbar(); cbar.set_label("Precision")
            plt.tight_layout()

        plt.show()

    interact(show_graph, view=["Macro-average", "Trend per Classe", "Heatmap"]);
# ! Caso LOS = True
# Hardcode risultati (21 valori per classe 0..20)

# ? TEST 1 = {Reso=0.5, STD=1, LOS=True}  [actual 4]
Test1 = [1.0000,0.4154,0.7341,0.6814,0.4196,0.7820,0.5341,0.5363,0.5271,0.5483,
         0.7228,0.8015,0.4033,0.6270,0.4064,1.0000,0.5387,0.3774,0.4905,0.6348,0.8110]

# ? TEST 2 = {Reso=2, STD=1, LOS=True} 
Test2 = [1.0000, 0.7793, 0.9566, 0.9171, 0.6910,
         0.9641, 0.9039, 0.8220, 0.7429, 0.8377,
         0.9307, 0.9669, 0.7648, 0.9271, 0.8047,
         1.0000, 0.7225, 0.6538, 0.8149, 0.9239,
         0.9790]

# ? TEST 3 = {Reso=0.5, STD=0.1, LOS=True} 
Test3 = [1.0000, 0.9730, 0.9964, 0.9902, 0.9617,
         0.9880, 0.9465, 0.9546, 0.9298, 0.9334,
         0.9770, 0.9961, 0.9633, 0.9615, 0.9269,
         1.0000, 0.9120, 0.9000, 0.9816, 0.9800,
         0.9941]

# ? TEST 4 = {Reso=2, STD=0.1, LOS=True} [actual 1]
Test4 = [1.0000, 0.9995, 0.9997, 0.9997, 0.9989,
         0.9999, 0.9996, 0.9993, 0.9959, 0.9984,
         0.9996, 1.0000, 0.9996, 0.9991, 0.9992,
         1.0000, 0.9972, 0.9964, 0.9998, 1.0000,
         0.9999]


tests = {"Test1 \n{Reso=0.5, STD=1, LOS=True}": Test1, "Test2 \n{Reso=2, STD=1, LOS=True}": Test2, "Test3 \n{Reso=0.5, STD=0.1, LOS=True}": Test3, "Test4 \n{Reso=2, STD=0.1, LOS=True}": Test4}
display_graph(tests, LOS=True)


# ! Caso LOS = False
# Hardcode risultati (21 valori per classe 0..20)

# ? TEST 1 = {Reso=0.5, STD=1, LOS=False}  [actual 8]
Test1 = [1.0000, 0.7021, 0.9594, 0.9371, 0.7134,
             0.9566, 0.9118, 0.8010, 0.7427, 0.7676,
             0.9563, 0.9638, 0.6891, 0.9365, 0.8406,
             1.0000, 0.7204, 0.7130, 0.7899, 0.9427,
             0.9734]
# ? TEST 2 = {Reso=2, STD=1, LOS=False} [actual 6]
Test2 = [1.0000, 0.9513, 0.9987, 0.9964, 0.9022,
             0.9983, 0.9967, 0.9445, 0.8724, 0.9345,
             0.9982, 0.9985, 0.9308, 0.9995, 0.9879,
             1.0000, 0.8686, 0.9205, 0.9499, 0.9989,
             0.9998]

# ? TEST 3 = {Reso=0.5, STD=0.1, LOS=False} [actual 7]
Test3 = [1.0000, 0.9996, 0.9999, 1.0000, 0.9985,
             0.9999, 0.9997, 0.9989, 0.9982, 0.9927,
             0.9999, 1.0000, 0.9984, 0.9990, 0.9993,
             1.0000, 0.9961, 0.9897, 0.9998, 0.9999,
             0.9999]

# ? TEST 4 = {Reso=2, STD=0.1, LOS=False} [actual 5]
Test4 = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 1.0000, 1.0000, 0.9999, 1.0000,
             1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000]

tests = {"Test1 \n{Reso=0.5, STD=1, LOS=False}": Test1, "Test2 \n{Reso=2, STD=1, LOS=False}": Test2, "Test3 \n{Reso=0.5, STD=0.1, LOS=False}": Test3, "Test4 \n{Reso=2, STD=0.1, LOS=False}": Test4}
display_graph(tests, LOS=False)
