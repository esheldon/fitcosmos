import numpy as np

def view_mbobs_list(mbobs_list, **kw):
    import biggles
    import images
    import plotting

    show = kw.get('show',False)

    weight=kw.get('weight',False)
    nband=len(mbobs_list[0])

    if weight:
        grid=plotting.Grid(len(mbobs_list))
        plt=biggles.Table(
            grid.nrow,
            grid.ncol,
        )
        aratio = grid.nrow/(grid.ncol*2)
        plt.aspect_ratio = aratio
        for i,mbobs in enumerate(mbobs_list):
            if nband==3:
                im=make_rgb(mbobs)
            else:
                im=mbobs[0][0].image
            wt=mbobs[0][0].weight

            row,col = grid(i)

            tplt=images.view_mosaic([im, wt], show=False)
            plt[row,col] = tplt

        if show:
            plt.show(width=2000, height=2000*aratio)
    else:
        if nband==6:

            grid=plotting.Grid(len(mbobs_list))
            plt=biggles.Table(
                grid.nrow,
                grid.ncol,
            )

            #if grid.nrow==1:
            #    plt.aspect_ratio = grid.nrow/grid.ncol
            #else:
            #plt.aspect_ratio = grid.nrow/(grid.ncol*2)
            plt.aspect_ratio = grid.nrow/grid.ncol

            for i,mbobs in enumerate(mbobs_list):
                des_im=make_rgb(mbobs[1:1+3])
                des_wt=mbobs[2][0].weight
                cosmos_im=mbobs[4][0].image
                cosmos_wt=mbobs[4][0].weight

                tplt=images.view_mosaic(
                    [cosmos_im, des_im,
                     cosmos_wt, des_wt],
                    titles=['cosmos','DES','cosmos wt', 'DES wt'],
                    show=False,
                )

                row,col = grid(i)
                plt[row,col] = tplt

        else:
            imlist=[mbobs[0][0].image for mbobs in mbobs_list]

            plt=images.view_mosaic(imlist, **kw)
    return plt

def compare_models(fofid, mbobs_list, output, fitter, **kw):
    import biggles
    import images
    import plotting

    show=kw.get('show',False)
    save=kw.get('save',False)

    for iobj,mbobs in enumerate(mbobs_list):
        id = output['id'][iobj]
        for band,obslist in enumerate(mbobs):
            for obsnum,obs in enumerate(obslist):
                model_image = fitter.make_image(
                    iobj,
                    band=band,
                    obsnum=obsnum,
                    include_nbrs=True,
                )

                title='fof: %d id: %d band: %d obs: %d' % (fofid, id, band,obsnum)

                image = obs.image
                #wt=obs.weight.copy()
                #wt *= 1.0/wt.max()
                #image = image * wt
                #model_image *= wt

                #images.compare_images(
                #    obs.image,
                #    model_image,
                #    labels=['image','model'],
                #    title=title,
                #)
                plt=compare_images_mosaic(
                    image,
                    model_image,
                    labels=['image','model'],
                    title=title,
                    show=show,
                )
                if save:
                    fname = 'compare-fof-%d-%d-band%d-%d.png' % (fofid,id,band,obsnum)
                    print(fname)
                    plt.write_img(1500,1500*2.0/3.0,fname)




def make_rgb(mbobs):
    import images

    #SCALE=.015*np.sqrt(2.0)
    SCALE=0.01
    # lsst
    #SCALE=0.0005
    #relative_scales = np.array([1.00, 1.2, 2.0])
    relative_scales = np.array([1.00, 1.0, 2.0])

    scales= SCALE*relative_scales

    r=mbobs[2][0].image
    g=mbobs[1][0].image
    b=mbobs[0][0].image

    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=0.12,
    )
    return rgb

def compare_images_mosaic(im1, im2, **keys):
    import biggles
    import copy
    import images

    show=keys.get('show',True)
    ymin=keys.get('min',None)
    ymax=keys.get('max',None)

    color1=keys.get('color1','blue')
    color2=keys.get('color2','orange')
    colordiff=keys.get('colordiff','red')

    nrow=2
    ncol=3

    label1=keys.get('label1','im1')
    label2=keys.get('label2','im2')

    cen=keys.get('cen',None)
    if cen is None:
        cen = [(im1.shape[0]-1)/2., (im1.shape[1]-1)/2.]

    labelres='%s-%s' % (label1,label2)

    biggles.configure( 'default', 'fontsize_min', 1.)

    if im1.shape != im2.shape:
        raise ValueError("images must be the same shape")


    #resid = im2-im1
    resid = im1-im2
    mval = min(im1.min(), im2.min(), resid.min())

    # will only be used if type is contour
    tab=biggles.Table(2,1)
    if 'title' in keys:
        tab.title=keys['title']

    tkeys=copy.deepcopy(keys)
    tkeys.pop('title',None)
    tkeys['show']=False
    tkeys['file']=None


    tkeys['nonlinear']=None
    # this has no effect
    tkeys['min'] = resid.min()
    tkeys['max'] = resid.max()

    mosaic = np.zeros( (im1.shape[0], 3*im1.shape[1]) )
    ncols = im1.shape[1]
    mosaic[:,0:ncols] = im1 - mval
    mosaic[:,ncols:2*ncols] = im2 - mval
    mosaic[:,2*ncols:3*ncols] = resid - mval

    residplt=images.view(mosaic, **tkeys)

    dof=im1.size
    chi2per = (resid**2).sum()/dof
    lab = biggles.PlotLabel(0.9,0.9,
                            r'$\chi^2/npix$: %.3e' % chi2per,
                            color='red',
                            halign='right')
    residplt.add(lab)



    cen0=int(cen[0])
    cen1=int(cen[1])
    im1rows = im1[:,cen1]
    im1cols = im1[cen0,:]
    im2rows = im2[:,cen1]
    im2cols = im2[cen0,:]
    resrows = resid[:,cen1]
    rescols = resid[cen0,:]

    him1rows = biggles.Histogram(im1rows, color=color1)
    him1cols = biggles.Histogram(im1cols, color=color1)
    him2rows = biggles.Histogram(im2rows, color=color2)
    him2cols = biggles.Histogram(im2cols, color=color2)
    hresrows = biggles.Histogram(resrows, color=colordiff)
    hrescols = biggles.Histogram(rescols, color=colordiff)

    him1rows.label = label1
    him2rows.label = label2
    hresrows.label = labelres
    key = biggles.PlotKey(0.1,0.9,[him1rows,him2rows,hresrows]) 

    rplt=biggles.FramedPlot()
    rplt.add( him1rows, him2rows, hresrows,key )
    rplt.xlabel = 'Center Rows'

    cplt=biggles.FramedPlot()
    cplt.add( him1cols, him2cols, hrescols )
    cplt.xlabel = 'Center Columns'

    rplt.aspect_ratio=1
    cplt.aspect_ratio=1

    ctab = biggles.Table(1,2)
    ctab[0,0] = rplt
    ctab[0,1] = cplt

    tab[0,0] = residplt
    tab[1,0] = ctab

    images._writefile_maybe(tab, **keys)
    images._show_maybe(tab, **keys)

    return tab

