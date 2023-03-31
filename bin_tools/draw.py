# -*- coding: utf-8 -*-  
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
plt.rcParams["font.sans-serif"]=['SimHei']#显示中文
import io
from PIL import Image
import xlsxwriter

## TODO
## save
def scatter_size_color(x,y,size,color,xtick, ytick, save=None,title = None, vmin=0,vmax=0.1):
    plt.figure(figsize=[len(xtick)*2,len(ytick)*2],dpi=100)
    org = cm.get_cmap('Oranges', 128)
    size = ((size * 10000) / size.max()).apply(lambda x:max(x, 100))
    plt.scatter(x,y,s=size,cmap=org,c=color, vmin = vmin, vmax = vmax)
    #plt.xlabel(i1+key_map.get(i1,""))
    #plt.ylabel(i2+key_map.get(i2,""))
    plt.xlim(xtick.index.min() - 0.5, xtick.index.max() + 0.5)
    plt.ylim(ytick.index.min() - 0.5, ytick.index.max() + 0.5)
    
    plt.xticks(xtick.index.tolist(),xtick.tolist(), rotation=45,ha="right")
    plt.yticks(ytick.index.tolist(),ytick.tolist(), rotation=45,ha="right")
    if not title is None:
        plt.title(title)
    if not save is None:
        plt.savefig(save, dpi = 400, bbox_inches = 'tight')


def draw_bar(_df, save = None, title = None, ax = None, draw_bad_rate: object = True):
    clr_label_dict = _df[["label", "part"]]. drop_duplicates().set_index("part")["label"]
    clr_dict_raw = {"1":'yellow', "2":'green', "3":"gray", "4": "orange"}
    clr_dict = {j:clr_dict_raw[i] for i, j in clr_label_dict.items()}
    font = {'family' : 'mono',
            'color'  : 'darkred',
            'weight' : 'normal',
            'size'   : 16}
    sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
    #plt.axes([0.1, 0.1, 0.9, 0.9], frameon=False, xticks=[],yticks=[])
    #error_params = dict(elinewidth=1,ecolor='coral',capsize=0)
    kwargs = dict()
    if ax is not None:
        kwargs["ax"] = ax

    g = sns.catplot(x = "bin",
                    y = "porp",
                    hue = "label",
                    data = _df,
                    height = 3,
                    kind = "bar",
                    palette = clr_dict,
                    saturation = 0.7,
                    aspect = 1.5,
                    alpha = 0.9,
                    legend = False,
                    **kwargs
                    )
    plt.xticks(rotation=45, size = 5)
    #plt.gca().set(ylim = (0, 1))
    g.set_yticklabels(size = 5)
    g.set_axis_labels("","PORPOTION")
    g.despine(left=True)
    
    _x_dict = {i._text:int(i._x) for i in g.axes[0][0].get_xticklabels()}
    _df["_x"] = _df["bin"]. astype(str).apply(lambda x:_x_dict.get(x))
    #_xticks = _df[["bin","code"]].drop_duplicates("code").sort_values("code")["bin"].tolist()
    #g.set_xticklabels(_xticks, rotation = 45,size = 5)

    if not title is None:
        plt.title(title, y = 0.9)   


    if draw_bad_rate is True:
        _df["bad_rate1"]=_df["bad_rate"]
        _plt = g.ax.twinx()
        _plt.grid(False)
        _ylim_top = _df["bad_rate1"]. max() * 2
        _plt.set_ylim(bottom = 0, top = _ylim_top)
        _plt.tick_params(axis='y', labelsize= 5)


        for i in list(set(_df["part"])):
            _label = _df[_df["part"]==i]["label"]. iloc[0]
            _plt.plot(_df["_x"][_df["part"]==i],_df["bad_rate1"][_df["part"]==i],linestyle="--",marker='*',c=clr_dict_raw[i], label = _label)
        #ax1.legend(loc = 'upper left', bbox_to_anchor=(1.1, 0.7))
        
    ax1 = g.ax
    ax1.legend(fontsize = 8)
    if not save is None:
        plt.savefig(save, dpi = 200, bbox_inches = 'tight')
    return None

def get_resized_image_data(file_path, bound_width_height = None):
    # get the image and resize it
    im = Image.open(file_path)
    if bound_width_height is not None:
        im.thumbnail(bound_width_height, Image.ANTIALIAS)  # ANTIALIAS is important if shrinking
    # stuff the image data into a bytestream that excel can read
    im_bytes = io.BytesIO()
    im.save(im_bytes, format='PNG')
    return im_bytes, im.size

def insert_image(worksheet, file, row, col, x_scale = 1, y_scale = 1):
    _pic, _size = get_resized_image_data(file)
    worksheet.set_column(col, col, 24.38 * x_scale)
    worksheet.set_row(row, 150 * y_scale)
    worksheet.insert_image(row, col, file, {'x_scale': 170 / _size[0] * x_scale, 'y_scale': 200 / _size[1] * y_scale, "image_data": _pic})

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    type(ax1)


    file = "./SMY_png/{0}". format(_pics[i])
    # 插入一张图片
    workbook = xlsxwriter.Workbook('images.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range(10):
        _i1, _i2 = int(i / 3), i % 3
        insert_image(worksheet, "./SMY_png/{0}". format(_pics[i]), row = 2 * _i1 + 1, col = 2 * _i2 + 1, x_scale = 3, y_scale = 2)

    workbook.close()

