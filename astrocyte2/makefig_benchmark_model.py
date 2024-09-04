import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

########## GIVE THE NAME OF THE COMPOSITE FIGURE TO BE GENERATED ##########
composite_figure_name_root = 'benchmark_model'

########## DEFINE PANELS AND THE FIGURE ##########
space = 0.5
panels_dict = {
    "benchmark_model_A": {"width": 4, "height": 3, "position": (0, 0.5+1.5+3+1.5+space*3)},
    "benchmark_model_B": {"width": 4, "height": 3, "position": (4, 0.5+1.5+3+1.5+space*3)},
    "benchmark_model_C": {"width": 4, "height": 1.5, "position": (0, 0.5+1.5+3+space*2)},
    "benchmark_model_D": {"width": 4, "height": 1.5, "position": (4, 0.5+1.5+3+space*2)},

    "benchmark_model_E": {"width": 4, "height": 3, "position": (0, 0.5+1.5+space*1)},
    "benchmark_model_F": {"width": 4, "height": 3, "position": (4, 0.5+1.5+space*1)},
    "benchmark_model_G": {"width": 4, "height": 1.5, "position": (0, 0.5)},
    "benchmark_model_H": {"width": 4, "height": 1.5, "position": (4, 0.5)},
}
final_panel_shrink = 1.0
figure_size_inch = (4+4, 0.5+1.5+3+1.5+3+space*4+0.2)

########## SET PANEL LABELS ##########
shift = 0.2
label_names = [
    [
        {"A": (0, 0.5+1.5+3+1.5+3+space*3+shift)},
        {"B": (0, 0.5+1.5+3+1.5+space*2+shift)},
        {"C": (0, 0.5+1.5+3+space+shift)},
        {"D": (0, 0.5+1.5+shift)},
    ],
]
label_text = [
    [
        {"Sparse":      (2,   figure_size_inch[1]-0.15)},
        {"Synchronous": (4+2, figure_size_inch[1]-0.15)},
        {"Raster plot": (3.8, 0.5+1.5+3+1.5+3+space*3+shift)},
        {"Neuronal firing rate": (3.8, 0.5+1.5+3+1.5+space*2+shift)},
        {"Astrocytic dynamics": (3.8, 0.5+1.5+3+space+shift)},
        {"Astrocytic SIC input per neuron": (3.8, 0.5+1.5+shift)},
        {"Time (ms)":    (2,   0.2)},
        {"Time (ms)":    (4+2, 0.2)},
    ],
]

# get figure dimensions in cm
cm_per_inch = 2.54
figure_size_cm = (figure_size_inch[0]*cm_per_inch, figure_size_inch[1]*cm_per_inch)
figure_left_cm = -figure_size_cm[0]/2
figure_bottom_cm = -figure_size_cm[1]/2

def panel_label(s, pos, size=15, bold=True, transform="figure", ha="left", va="center"):
    ax = plt.gca()
    fig = plt.gcf()
    transform_object = fig.dpi_scale_trans if transform == "figure" else ax.transAxes
    if bold:
        plt.text(pos[0],pos[1],r'\bfseries{}%s' % s,transform=transform_object,ha=ha,va=va,size=size)
    else:
        plt.text(pos[0],pos[1],r'%s' % s,transform=transform_object,ha=ha,va=va,size=size)

    return 0

def some_matplotlib_figure(
        fig_size=(8, 4),             ## figure size (width, height) in inches
        dpi=400,                     ## print resolution
        subtitles=[],
        subtitle_size=8,
        name_root="",
    ):

    plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.size']= font_size
    plt.figure(1,figsize=fig_size,dpi=dpi)
    plt.clf()

    # Do it without subplots
    plt.axis("off")

    def function_a(list_in, size, bold, ha, va):
        for x in list_in:
            for y in x:
                try:
                    panel_label(list(y.keys())[0], list(y.values())[0], size=size, bold=bold, ha=ha, va=va)
                except:
                    pass

    function_a(label_names, size=20, bold=True, ha="left", va="center")
    function_a(label_text, size=20, bold=False, ha="center", va="center")
    function_a(subtitles, size=subtitle_size, bold=False, ha="center", va="center")

    fname = 'master_figure'
    plt.savefig("%s.pdf" % fname)
    plt.savefig("%s.eps" % fname)

    return fname, fig_size

def create_composite_figure(name_root,master_file_name,panels_dict,draw_grid=False,fig_size=(10,10),folder_name="make_figure_benchmark_model"):

    file = open('%s.tex' % name_root , 'w')
    file.write(r"\documentclass{article}")
    file.write("\n")
    file.write(r"\usepackage{geometry}")
    file.write("\n")
    file.write(r"\geometry{paperwidth=%.3fin, paperheight=%.3fin, top=0pt, bottom=0pt, right=0pt, left=0pt}" % (fig_size[0],fig_size[1]))
    file.write("\n")
    file.write(r"\usepackage{tikz}")
    file.write("\n")
    file.write(r"\usepackage{graphicx}")
    file.write("\n")
    file.write(r"\pagestyle{empty}")
    file.write("\n")
    file.write(r"\begin{document}")
    file.write("\n")
    file.write(r"\noindent")
    file.write("\n")
    file.write(r"  \begin{tikzpicture}%")
    file.write("\n")
    file.write(r"    \node[inner sep=-1pt] (matplotlib_figure) at (0,0)")
    file.write("\n")
    file.write(r"    {\includegraphics{%s}};" % (master_file_name))
    file.write("\n")
    for key, value in panels_dict.items():
        file_path = os.path.join(folder_name, key) if isinstance(folder_name, str) else key
        if not os.path.isfile(file_path + ".eps"):
            print(f"{file_path + '.eps'} does not exist!")
            continue
        position_cm = (figure_left_cm+value["position"][0]*cm_per_inch,figure_bottom_cm+value["position"][1]*cm_per_inch)
        file.write(r"    \node[inner sep=-1pt,rectangle,anchor=south west] (inkscape_sketch) at (%.4f,%.4f)" % position_cm)
        file.write("\n")
        file.write(r"    {\includegraphics[width=%.2fcm]{%s}};" % (value["width"]*cm_per_inch*final_panel_shrink, file_path))
        file.write("\n")
    file.write(r"  \end{tikzpicture}%")
    file.write("\n")
    file.write(r"\end{document}")
    file.write("\n")

    file.close()

    ## generate eps of composite figure
    os.system('latex %s.tex; dvips -o %s.eps %s.dvi' % (name_root,name_root,name_root))

    return 0

########################################

def run_all():
        ## create a matplotlib master figure and save as pdf (and eps)
        # (this figure defines the layout of the final composite figure)
        master_file_name, fig_size_out = some_matplotlib_figure(
            fig_size=figure_size_inch,
            name_root=composite_figure_name_root,
        )

        ## create the composite figure
        create_composite_figure(
            composite_figure_name_root,
            master_file_name,
            panels_dict,
            fig_size=fig_size_out
        )

        ## clean files
        os.system("rm *.log *.aux  *.dvi")

run_all()
