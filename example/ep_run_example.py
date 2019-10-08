"""
Shows an example of how to use run_energyplus functions

@author Carlos Duarte <cduarte@berkeley.edu>
"""

import glob
import sys
from os.path import join

sys.path.insert(0, join('../')) # path to run_energyplus
from run_energyplus import run_ep, run_rveso


if __name__ == '__main__':
    EP = "EnergyPlusV9-1-0"
    idf_file = [join("./", "2e5e45c0-347c-11e9-acb2-38b1db76d8cc.idf")]
    weather_file = [join("USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw")]
    outfolder = join("./", "Output")

    # Run energyplus
    run_ep(idf_file, weather_file, outfolder, EP=EP)

    # Extract variables from eso files
    esos = glob.glob(join(outfolder, "*.eso"))
    vrs = [
        'Zone Operative Temperature', 
        'Zone Radiant HVAC Mass Flow Rate']

    run_rveso(esos, vrs, join(outfolder), EP=EP)