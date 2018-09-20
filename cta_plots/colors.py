from itertools import cycle

LST = '#306fb0'
MST = '#6199ac'
SST = '#b2d2d4'

telescope_color = {
    'LST': LST,
    'MST': MST,
    'SST': SST,
}

telescope_color_monochrome = {
    'LST': '#808488',
    'MST': '#a2a7a7',
    'SST': '#d3d5d5'
}

telescope_color_complementary = {
    'LST': '#984d4d',
    'MST': '#b67777',
    'SST': '#c39db3'
}


main_color = '#4386dd'
main_color_complement = '#d63434'
dark_main_color = '#707070'

color_pallete = [
    '#cc2a36',
    '#4f372d',
    '#00a0b0',
    '#edc951',
    '#4ab174',
    '#eb6841',
]

color_cycle = cycle(color_pallete)

default_cmap = 'RdPu'
