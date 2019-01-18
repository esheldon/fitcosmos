import numpy as np

def view_mbobs_list(mbobs_list, **kw):
    import biggles
    import images
    import plotting

    weight=kw.get('weight',False)
    nband=len(mbobs_list[0])

    if weight:
        grid=plotting.Grid(len(mbobs_list))
        plt=biggles.Table(
            grid.nrow,
            grid.ncol,
        )
        for i,mbobs in enumerate(mbobs_list):
            if nband==3:
                im=make_rgb(mbobs)
            else:
                im=mbobs[0][0].image
            wt=mbobs[0][0].weight

            row,col = grid(i)

            tplt=images.view_mosaic([im, wt], show=False)
            plt[row,col] = tplt

        plt.show()
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

def compare_models(mbobs_list, fitter, **kw):
    import biggles
    import images
    import plotting

    for iobj,mbobs in enumerate(mbobs_list):
        for band,obslist in enumerate(mbobs):
            for obsnum,obs in enumerate(obslist):
                model_image = fitter.make_image(
                    iobj,
                    band=band,
                    obsnum=obsnum,
                )

                title='ind: %d band: %d obs: %d' % (iobj,band,obsnum)
                images.compare_images(
                    obs.image,
                    model_image,
                    labels=['image','model'],
                    title=title,
                )

    return

    weight=kw.get('weight',False)
    nband=len(mbobs_list[0])

    if weight:
        grid=plotting.Grid(len(mbobs_list))
        plt=biggles.Table(
            grid.nrow,
            grid.ncol,
        )
        for i,mbobs in enumerate(mbobs_list):
            if nband==3:
                im=make_rgb(mbobs)
            else:
                im=mbobs[0][0].image
            wt=mbobs[0][0].weight

            row,col = grid(i)

            tplt=images.view_mosaic([im, wt], show=False)
            plt[row,col] = tplt

        plt.show()
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


