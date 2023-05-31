import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from matplotlib.pyplot import MultipleLocator
cats = {}
colormapping = {}
atom_list = ["LLM"]

def get_loc(data):
    dets = [5,10,15,20,30,60,120,300,600]
    # 根据时间的跨度确定横坐标的刻度
    early = last = None
    for d in data:
        early = d[0] if early == None else min(early,d[0])
        last = d[1] if last == None else max(early,d[1])
    all_seconds = (last-early).total_seconds()
    det = all_seconds/8
    for x in dets:
        if det < x:
            det = x
            break
    det = min(det,dets[-1])
    locator = 0 if det < 60 else 1 # 0为秒数，1为分钟
    intervals = []
    idx = 0
    if locator > 0:
        det /= 60
    while(True):
        intervals.append(idx)
        idx += det
        if idx>=60:
            break
    loc = mdates.SecondLocator(bysecond=intervals) if locator == 0 \
        else mdates.MinuteLocator(byminute=intervals)
    return loc

def draw_timeline(ax,data,verts,colors):
    bars = PolyCollection(verts, facecolors=colors)
    ax.add_collection(bars)
    ax.autoscale()
    loc = get_loc(data)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    y_major_locator=MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    y_ticks = list(cats.values())
    y_labels = list(cats.keys())
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)

def draw_pie(ax,data):
    labels = []
    colors = []
    vals = []
    exists = {}
    atom_dict = {}

    idx = 0
    for d in data:
        if d[2] in atom_list and d[2] not in exists:
            exists[d[2]] = 1
    for atom in atom_list:
        if atom not in exists:
            continue
        labels.append(atom)
        colors.append(colormapping[atom])
        vals.append(0)
        atom_dict[atom] = idx
        idx += 1

    early = last = None
    atom_total_seconds = 0
    for d in data:
        early = d[0] if early == None else min(early,d[0])
        last = d[1] if last == None else max(early,d[1])
        if d[2] in atom_list:
            seconds = (d[1]-d[0]).total_seconds()
            idx = atom_dict[d[2]]
            vals[idx] += seconds
            atom_total_seconds += seconds
    all_seconds = (last-early).total_seconds()
    res_seconds = all_seconds - atom_total_seconds
    labels.append("ELSE")
    # res_seconds = 40
    vals.append(res_seconds)
    colors.append("C88")
    ax.pie(vals,labels=labels,colors=colors,autopct='%.2f%%')
    return 

def draw(data,savename="test.png"):
    for d in data:
        if d[2] not in cats:
            cats[d[2]] = len(cats)+1
            colormapping[d[2]] = "C"+str(len(cats))
    verts = []
    colors = []
    for d in data:
        v = [(mdates.date2num(d[0]), cats[d[2]]-0.48),
          (mdates.date2num(d[0]), cats[d[2]]+0.48),
          (mdates.date2num(d[1]), cats[d[2]]+0.48),
          (mdates.date2num(d[1]), cats[d[2]]-0.48),]
        verts.append(v)
        colors.append(colormapping[d[2]])
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(12,5))
    
    # plt.rcParams['xtick.labelsize'] = 10
    draw_timeline(ax[0],data,verts,colors)
    draw_pie(ax[1],data)
    plt.savefig(savename)
    plt.show()

