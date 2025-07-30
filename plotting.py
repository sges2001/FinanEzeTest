import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils

def vertical_bar(axs,compras,ccompras):
    ''' Plot the entrada y salida de posiciones '''
    start_indices, end_indices= utils.calc_startend(compras[:,None])
    start_cindices, end_cindices= utils.calc_startend(ccompras[:,None])

    indices=np.arange(compras.shape[0])
    for ax in axs:
        for start, end in zip(start_indices[0], end_indices[0]):
            ax.axvspan(indices[start], indices[end], alpha=0.3, color='green')
        for start, end in zip(start_cindices[0], end_cindices[0]):
            ax.axvspan(indices[start], indices[end], alpha=0.3, color='red')
    
def plot_zscore(j,res0,fname):
    nt=res0.spread.shape[1]

    res = copy.deepcopy(res0) 
    res.reorder(j) # select the pair

    figfile=fname+f'zscore{j}.png'
    fig = plt.figure(figsize=(7, 5))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res.assets[0],label=res.company[0])
    ax1.plot(res.assets[1],label=res.company[1])
    ax1.legend()
    ax1.set_title('Assets')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(nt),res.spread)
    ax2.plot(range(nt),res.spread_mean)
    ax2.fill_between(range(nt), res.spread_mean - 1.96* res.spread_std,
                     res.spread_mean +1.95* res.spread_std,color='gray', alpha=0.2)
    ax2.set_title('Spread')

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(res.zscore)
    ax3.set_title('Z-score')

    vertical_bar([ax3],res.compras,res.ccompras)
    
    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()


def plot_capital_single(j,res0,fname):
    nt=res0.spread.shape[1]
    res = copy.deepcopy(res0) 
    res.reorder(j) # select the pair
    
    figfile=fname+f'capital{j}.png'
    
    fig, ax = plt.subplots(3,1,figsize=(7,7))
    ax[0].plot(res.zscore)
    ax[0].set_title('Z-score')
    

    for ivar in range(res.corto.shape[-1]):
        ax[1].plot(res.corto[:,ivar],label='corto '+res.company[ivar])
    ax[1].legend()

    for ivar in range(res.corto.shape[-1]):
        ax[2].plot(res.largo[:,ivar],label='largo '+res.company[ivar])
    ax[2].legend()

    vertical_bar(ax,res.compras,res.ccompras)

    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()



def plot_capital_from_list(res_list, par_idx, fname):
    """
    Grafica la evolución del capital para el par `par_idx` a lo largo de múltiples iteraciones.
    
    Parámetros:
        res_list : lista de objetos res
        par_idx : índice del par (ej: 42)
        fname : prefijo del nombre de archivo
    """
    # Concatenar los datos del par seleccionado
    zscore_all = np.concatenate([res.zscore[par_idx] for res in res_list])
    print(zscore_all)
    print("zscore plottt",zscore_all.shape)
    spread_all = np.concatenate([res.spread[par_idx] for res in res_list])
    print(spread_all)
    print("spread plottt",spread_all.shape)

    corto_all = np.concatenate([res.corto[par_idx] for res in res_list])
    print(corto_all)
    print("corto plottt",corto_all.shape)

 
    largo_all = np.concatenate([res.largo[par_idx] for res in res_list])
    print(largo_all)
    print("largoplottt",largo_all.shape)
 
 
    compras_all = np.concatenate([res.compras[par_idx] for res in res_list])
    print(compras_all)
    print("comprasplottt",compras_all.shape)

    ccompras_all = np.concatenate([res.ccompras[par_idx] for res in res_list])
    print(ccompras_all)
    print("ccompras plottt",ccompras_all.shape)

    # Extraer info del primer objeto (los nombres de las compañías del par)
    res0 = res_list[0]
    company_pair = [res0.company[par_idx][0], res0.company[par_idx][1]]

    # Crear figura
    figfile = fname + f'capital_pair{par_idx}.png'
    fig, ax = plt.subplots(3, 1, figsize=(7, 7))

    ax[0].plot(zscore_all)
    ax[0].set_title('Z-score')

    for i in range(corto_all.shape[1]):
        ax[1].plot(corto_all[:, i], label='corto ' + company_pair[i])
    ax[1].legend()

    for i in range(largo_all.shape[1]):
        ax[2].plot(largo_all[:, i], label='largo ' + company_pair[i])
    ax[2].legend()

    # Agregar las barras verticales de compras y ventas
    vertical_bar(ax, compras_all, ccompras_all)

    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()
