import numpy as np 
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl 


class viz:
    '''Define the default visualize configure
    '''
    # -----------  Palette 1 -------------

    dBlue   = np.array([ 56,  56, 107]) / 255
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    lBlue2  = np.array([166, 201, 222]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    lGreen  = np.array([242, 251, 238]) / 255
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255
    lRed2   = np.array([254, 177, 175]) / 255
    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    ocGreen = np.array([ 90, 196, 164]) / 255
    Gray    = np.array([163, 161, 165]) / 255

    Palette = [Blue, Red, Yellow, ocGreen, Purple, Gray]

    # -----------  Palette 2 ------------- 

    dGreen  = np.array([ 15,  93,  81]) / 255
    llBlue  = np.array([118, 193, 202]) / 255
    Ebony   = np.array([ 86,  98,  70]) / 255
    deBlue  = np.array([ 66,  96, 118]) / 255
    fsGreen = np.array([ 79, 157, 105]) / 255
    Ercu    = np.array([190, 176, 137]) / 255
    ubSilk  = np.array([232, 204, 191]) / 255
    ppPant  = np.array([233, 214, 236]) / 255

    Palette2 = [Ebony, fsGreen, Yellow, ubSilk, Ercu, ppPant]

    # -----------  Palette 2 -------------  

    bOrange = np.array([222, 110,  75]) / 255
    deBrown = np.array([122, 101,  99]) / 255
    black   = np.array([  0,   0,   0]) / 255
    Palette3 = [bOrange, Blue, deBrown, ocGreen]

    # -----------  Divergence Palette -------------  

    r1 = np.array([248, 150,  30]) / 255
    r2 = np.array([249, 132,  74]) / 255
    r3 = np.array([249, 199,  79]) / 255
    g1 = np.array([144, 190, 109]) / 255
    g2 = np.array([ 67, 170, 139]) / 255
    g3 = np.array([ 77, 144, 142]) / 255
    divPalette = [r1, r2 ,r3, g1, g2, g3]

    # ----------- Color pairs ------------- 

    r1 = Red
    r2 = np.array([235, 179, 169]) / 255
    RedPairs  = [r1, r2]
    
    b1 = np.array([ 14, 107, 168]) / 255
    b2 = np.array([166, 225, 250]) / 255
    BluePairs = [b1, b2]

    p1 = np.array([142,  65,  98]) / 255
    p2 = np.array([237, 162, 192]) / 255
    PurplePairs = [p1, p2, r1]

    YellowPairs = [dYellow, Yellow]

    # -----------  Colormap ------------- 

    BluePalette   = [dBlue, Blue, lBlue]
    RedPalette    = [dRed, Red, lRed]
    YellowPalette = [dYellow, Yellow, lYellow]

    BluesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue, Blue])
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed, dRed])
    YellowsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow, Yellow])
    GreensMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizGreens',  [lGreen, Green])
    PurplesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizPurples', [np.clip(Purple*1.8, 0, 1), Purple])
    BluesMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue2, Blue])
    RedsMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed2, dRed])
    YellowsMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow2, Yellow])

    @staticmethod
    def get_style(): 
        # Larger scale for plots in notebooks
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Arial"

    @staticmethod
    def default_img_set():
    
        mpl.rcParams['interactive'] = True
        # figure
        mpl.rcParams['figure.frameon'] = False
        mpl.rcParams['figure.titleweight'] = 'regular'
        mpl.rcParams['figure.titlesize'] = 'xx-large'
        mpl.rcParams['figure.autolayout'] = True
        mpl.rcParams['figure.facecolor'] = 'w'
        mpl.rcParams['figure.edgecolor'] = 'None'
        mpl.rcParams['figure.constrained_layout.use'] = False
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'pdf'
        mpl.rcParams['savefig.facecolor'] = 'None'
        mpl.rcParams['savefig.edgecolor'] = 'None'
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'

        # axes
        mpl.rcParams['axes.facecolor'] = 'None'
        mpl.rcParams['axes.labelweight'] = 'bold'
        mpl.rcParams['axes.labelsize'] = 'x-large'
        mpl.rcParams['axes.titleweight'] = 'regular'
        mpl.rcParams['axes.titlesize'] = 'xx-large'
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['xtick.major.size'] = 5
        mpl.rcParams['xtick.labelsize'] = 'large'
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['ytick.major.size'] = 5
        mpl.rcParams['ytick.labelsize'] = 'large'
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        # Character
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.weight'] = 'bold'
        # legend
        mpl.rcParams['legend.frameon'] = False
            