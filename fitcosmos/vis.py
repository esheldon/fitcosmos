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
        if nband==3:
            imlist=[make_rgb(mbobs) for mbobs in mbobs_list]
        else:
            imlist=[mbobs[0][0].image for mbobs in mbobs_list]

        plt=images.view_mosaic(imlist, **kw)
    return plt

def make_rgb(mbobs):
    import images

    #SCALE=.015*np.sqrt(2.0)
    #SCALE=0.001
    # lsst
    SCALE=0.0005
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


