import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.ticker as ticker

def viz_IL_reId(styles, fig_labels, x, all_data, saved=False, save_path='saved.pdf'):
    '''
    [Example] of a style:
    style_default = {
            'label': 'k-same',
            'color': 'k',
            'marker': 's',
            'linestyle': '--',
            'markersize': 4,
            'linewidth': 1,
            'xlabel':'k Value',
            'xticks': np.arange(x[0], x[-1]+1, 2.0)
        }
    [Example] of a fig_label:
            fig_labels_0 = {
            'title': '',
            'ylabel':'Information Loss',
            'legendLoc': 0,
        }
    '''
    assert len(all_data) == len(fig_labels)
    assert len(all_data[0]) == len(styles)

    n_figs = len(all_data)
    fig, ax = plt.subplots(nrows=1, ncols=n_figs, figsize=(4*n_figs,3))
    
    for index, fig_data in enumerate(all_data):
        if len(all_data) == 1:
            style=styles[0]
            data = fig_data[0]
            ax.plot(x, data,
                               label=style['label'],
                               color=style['color'],
                               marker=style['marker'],
                               linestyle=style['linestyle'],
                               linewidth=style['linewidth'],
                               markersize=style['markersize']
                              )
            ax.set_ylabel(fig_labels[index]['ylabel'])
            ax.set_xlabel(style['xlabel'])
            ax.legend(loc=fig_labels[index]['legendLoc'])
            ax.set_xticks(style['xticks'])
            ax.set_title(fig_labels[index]['title'])
        else: 
            for style, data in zip(styles, fig_data):
                ax[index].plot(x, data,
                               label=style['label'],
                               color=style['color'],
                               marker=style['marker'],
                               linestyle=style['linestyle'],
                               linewidth=style['linewidth'],
                               markersize=style['markersize']
                              )
                ax[index].set_ylabel(fig_labels[index]['ylabel'])
                ax[index].set_xlabel(style['xlabel'])
                ax[index].legend(loc=fig_labels[index]['legendLoc'])
                ax[index].set_xticks(style['xticks'])
                ax[index].set_title(fig_labels[index]['title'])

    # ax[1].plot(x, [1/y for y in x], label = '1/k', color='gray', linestyle='-.', linewidth=1, markersize=5)
    fig.tight_layout()

    if saved:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def show_avg_dist(labelList, fakeid_dist_avg, filename = ''):
    fig, ax = plt.subplots()
    #plt.style.use('classic')
    plt.bar(labelList, fakeid_dist_avg, color='gray')
    plt.axhline(y=0.6, color='r', linestyle='-')
    plt.ylim(0.3,0.9)
    plt.grid(linestyle='-.', linewidth='0.5')
    plt.title('Average distance from clusters to fake ID')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average distance')
    if not filename == '':
        plt.savefig('Outputs/' + filename,dpi=300, bbox_inches = "tight")
    plt.show()

def show_boxplot_dist(all_distances, filename = ''):
    fig, ax = plt.subplots()
    bplot = plt.boxplot([x for x in all_distances], patch_artist=True)
    plt.axhline(y=0.6, color='r', linestyle='-')

    for components in bplot.keys():
        for line in bplot[components]:
            line.set_color('black')

    colors = ['gray']*len(all_distances)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for median in bplot['medians']:
        median.set(color='yellow')
    
    plt.title('Distance from fake ID to all cluster members')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Distance')
    #ax.set_aspect(1.5)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid(linestyle='-.', linewidth='0.5')
    if not filename == '':
        plt.savefig('Outputs/' + filename,dpi=300, bbox_inches = "tight")
    plt.show()

def show_avg_dist(labelList, fakeid_dist_avg, filename = ''):
    fig, ax = plt.subplots()
    #plt.style.use('classic')
    plt.bar(labelList, fakeid_dist_avg, color='gray')
    plt.axhline(y=0.6, color='r', linestyle='-')
    plt.ylim(0.3,0.9)
    plt.grid(linestyle='-.', linewidth='0.5')
    plt.title('Average distance from clusters to fake ID')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average distance')
    if not filename == '':
        plt.savefig('Outputs/' + filename,dpi=300, bbox_inches = "tight")
    plt.show()
    
    

def show_boxplot_dist(all_distances, filename = ''):
    fig, ax = plt.subplots()
    bplot = plt.boxplot([x for x in all_distances], patch_artist=True)
    plt.axhline(y=0.6, color='r', linestyle='-')

    for components in bplot.keys():
        for line in bplot[components]:
            line.set_color('black')

    colors = ['gray']*len(all_distances)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for median in bplot['medians']:
        median.set(color='yellow')
    
    plt.title('Distance from fake ID to all cluster members')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Distance')
    #ax.set_aspect(1.5)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid(linestyle='-.', linewidth='0.5')
    if not filename == '':
        plt.savefig('Outputs/' + filename,dpi=300, bbox_inches = "tight")
    plt.show()

def show_cluster_ids(cluster_id, cluster_index_dict, raw_data):
  fig=plt.figure(figsize=(5, 5), dpi=160)
  columns = 5
  rows = math.ceil(len(cluster_index_dict[cluster_id])/columns)
  for i, index in enumerate(cluster_index_dict[cluster_id]):
      fig.add_subplot(rows, columns, i+1)
      img = plt.imread(raw_data[index]['imagePath'])
      #print(raw_data[index]['imagePath'])
      #identity = int(raw_data[index]['imagePath'].split('_')[1])
      plt.imshow(img)
      plt.axis('off')
      #plt.title('ID %d'%identity)
  plt.show()

# def multiple_plot(x,y,n_plot=1, n_fig=1, labels = ['']*n_plot, color=['k']*n_plot, marker=['o']*n_plot, linestyle=['-']*n_plot,
#             titles=['']*n_fig, ylabels=['']*n_fig, legendLoc=[0]*n_fig
#             ):
#     '''
#     Input format:
#     {'x_data': x,'y_data': y, 'label': label, 'color': 'k', 'marker':'o', 'linestyle': '-'}
#     {'title': '', 'ylabel': '', 'legendLoc': 0}
#     '''
    


