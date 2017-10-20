import numpy as np
from ..antechamber.atomtype import atomtype
import math


def pdb(molecule, ff='amber', fn='cnt.pdb', save_dir='.'):
    """Creates a .pdb (protein data bank) type list for use with molecular dynamics packages.
    pdb_lines returns list holding every line of .pdb file."""
    amber_2_opls = {"CA": "CA", "HC": "HA"}
    atom_lines = []
    conect_lines = []
    inc_index_bondList = molecule.bondList + 1  # sets index to same index as "serial", works
    conect_header_bare = "CONECT"
    for i in range(len(molecule.posList)):
        serial = i + 1
        if ff == 'amber':
            name = molecule.atomtypes[i]
        elif ff == 'oplsaa':
            name = amber_2_opls[molecule.atomtypes[i]]
        else:
            print('Check ff input')
            raise SystemExit
        altloc = " "
        #resname = "CNT"
        resname = name
        chainid = "A"
        resseq = serial
        icode = " "
        x = round(molecule.posList[i][0], 3)
        y = round(molecule.posList[i][1], 3)
        z = round(molecule.posList[i][2], 3)
        occupancy = 1.00
        tempfactor = 0.00
        element = atomtype.inv_atomicSymDict[molecule.zList[i]]  # atomic number
        charge = " "
        atom_header = "ATOM  {0:>5}  {1:<3}{2}{3:>3} {4}{5:>4}{6}   {7:>8.3f}{8:>8.3f}{9:>8.3f}{10:>6.2f}{11:>6.2f}   " \
                      "       {12:>2}{13:>2}".format(serial, name, altloc, resname,
                                                     chainid, resseq, icode, x, y, z,
                                                     occupancy, tempfactor, element, charge)
        atom_lines.append(atom_header)
    for j in range(len(inc_index_bondList)):
        conect_header_temp = conect_header_bare
        for k in range(len(inc_index_bondList[j])):  # builds variable size conect header
            conect_adder = "{0:>5}".format(inc_index_bondList[j][k])
            conect_header_temp += conect_adder
        conect_lines.append(conect_header_temp)
    pdb_lines = atom_lines + conect_lines
    pdb_lines.append("TER")
    pdb_lines.append("END")
    save_file(pdb_lines, save_dir, fn)
    print('Successfully exported %s to %s' % (fn, save_dir))


def gro(molecule, scale=2.0, fn='cnt.gro', save_dir='.', periodic=False):
    """Creates a .gro file for gromacs, holds atom coordinates and unit cell size
    Coordinates exported in nm, originally in angstroms"""
    gro_lines = []
    res_num = 1
    res_name = "CNT"
    elemtypes = []
    a_num = []
    for i in range(len(molecule.atomtypes)):
        a_num_temp = atomtype.inv_atomicSymDict[molecule.zList[i]]  # get atomic number
        elemtypes_temp = molecule.atomtypes[i][0]  # element only (first char.)
        a_num.append(a_num_temp)
        elemtypes.append(elemtypes_temp)
    dist_to_orig = []
    for i in range(len(molecule.posList)):
        temp_dist = np.sqrt(molecule.posList[i][0]**2 + molecule.posList[i][1]**2 + molecule.posList[i][2]**2)
        dist_to_orig.append(temp_dist)
    index_min = np.argmin(dist_to_orig)  # closest pt in object to origin
    # move tube to true origin at 0,0,0
    move_dist = np.abs(molecule.posList[index_min])
    posList_cent = molecule.posList
    #posList_cent += move_dist  # now centered at origin
    x_list, y_list, z_list = zip(*molecule.posList)
    # center tube in quadrant 1 box
    max_x = np.max(x_list)
    max_y = np.max(y_list)
    max_z = np.max(z_list)
    min_x = np.min(x_list)
    min_y = np.min(y_list)
    min_z = np.min(z_list)
    length_x = np.abs(max_x-min_x)
    length_y = np.abs(max_y-min_y)
    length_z = np.abs(max_z-min_z)
    dist_to_move = np.abs([min_x, min_y, min_z])
    max_length = np.max([length_x, length_y, length_z])
    idx_max_dim = np.argmax([length_x, length_y, length_z])
    dims = ['X','Y','Z']
    dims_dict = {0:x_list, 1:y_list, 2:z_list}
    length_str = dims[idx_max_dim]
    print('Length of tube is in the %s direction.' % length_str)
    box_dim = scale * max_length
    new_move_dist = box_dim/2.0
    # measure dist to new box origin in move in every direction
    posDist_new = posList_cent
    posDist_new += (new_move_dist + dist_to_move)
    # now tube is centered in quadrant 0 box
    if periodic:
        dir_to_cut = dims_dict[idx_max_dim]
        left = np.min(dir_to_cut)  # picks first smallest value in list
        right = np.max(dir_to_cut)
        split = (right-left) / 2.0
        print('Splitting %s direction around %.2f' % (length_str, split))
        # find an atom closest to split line
        dist_to_split = []
        for i in range(len(molecule.posList)):
            temp_dist = np.abs(posDist_new[i][idx_max_dim] - split)
            dist_to_split.append(temp_dist)
        index_split = np.argmin(dist_to_split)  # closest pt in object to split


    # scale everything by bond
    posDist_new *= 0.1
    box_dim *= 0.1
    print("Box with be %dX the maximum dimension of the object.\nUsing a %.2fX%.2fX%.2f box." %
          (scale, box_dim, box_dim, box_dim))
    # lets write to list
    tit = "SWCNT armchair"
    tot_atoms = len(posDist_new)
    num_atoms_line = "{0:>5}".format(tot_atoms)
    gro_lines.append(tit)
    gro_lines.append(num_atoms_line)
    for i in range(len(posDist_new)):
        _index = i + 1
        temp_dist = np.sqrt(posDist_new[i][0] ** 2 + posDist_new[i][1] ** 2 + posDist_new[i][2] ** 2)
        temp_line = "{0:>5}{1:<5}{2:>5}{3:>5}{4:>8.3f}{5:>8.3f}{6:>8.3f}"\
            .format(res_num, res_name, a_num[i], _index,
                    posDist_new[i][0], posDist_new[i][1], posDist_new[i][2])
        gro_lines.append(temp_line)
    box_line = "{0:>8.3f}{1:>8.3f}{2:>8.3f}".format(box_dim, box_dim, box_dim)
    gro_lines.append(box_line)
    save_file(gro_lines, save_dir, fn)
    print('Successfully exported %s to %s' % (fn, save_dir))


def restrains(mol, fn='posre.itp', save_dir='.', fc=1000):
    """Generates posre.itp file used by GROMACS to restrain atoms to a location, can be read by x2top"""
    # force constant of position restraint (kJ mol^-1 nm^-2)
    # IF NEEDED. REMOVING CM TRANSLATION AND ROTATION IS PROBABLY BEST.
    # **********MAKE SURE MOLECULE IS HYDROGENATED FIRST********** #

    itp_lines = []
    funct = 1
    itp_lines.extend(["; file for defining restraints in CNT, read in through X.top", ""])
    itp_lines.extend(["[ position_restraints ]", "; ai  funct  fcx    fcy    fcz"])
    for i in mol.hcap:
        index = i + 1
        temp_line = "{0:>4}{1:>6}{2:>9}{3:>8}{4:>8}".format(index, funct, fc, fc, fc)
        itp_lines.append(temp_line)
    itp_lines.append("")  # EOL
    save_file(itp_lines, save_dir, fn)
    print('Successfully exported %s to %s' % (fn, save_dir))


def lammps(molecule, fn='cnt.lammps', save_dir='.', type_list=None):
    """Generates data file for use in LAMMPS
    Assuming 'real' units (the unit type)
    mass = grams/mole
    distance = Angstroms
    time = femtoseconds
    energy = Kcal/mole
    velocity = Angstroms/femtosecond
    force = Kcal/mole-Angstrom
    torque = Kcal/mole
    temperature = Kelvin
    pressure = atmospheres
    dynamic viscosity = Poise
    charge = multiple of electron charge (1.0 is a proton)
    dipole = charge*Angstroms
    electric field = volts/Angstrom
    density = gram/cm^dim
    bond_const_K =  Kcal/(mole*Angstrom^2)
    bond_const_r0 = Angstrom

    """
    vdwDict = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.8, 16: 1.8, 17: 2.75}
    amuDict = {1: 1.008, 6: 12.01, 7: 14.01, 8: 16.00, 9: 19.00,
               15: 30.79, 16: 32.065, 17: 35.45}
    if type_list is None:
        type_list = np.ones(len(molecule.posList))
    l_lines = []
    l_lines.append('LAMMPS Description')
    l_lines.append('')
    l_lines.append('%d atoms' % len(molecule.posList))
    l_lines.append('%d bonds' % len(molecule.bondList))
    l_lines.append('%d angles' % len(molecule.angleList))
    l_lines.append('%d dihedrals' % 0)
    l_lines.append('%d impropers' % 0)
    l_lines.append('')
    l_lines.append('%d atom types' % len(molecule.zList))
    l_lines.append('%d bond types' % len(molecule.bondList))
    l_lines.append('%d angle types' % len(molecule.angleList))
    l_lines.append('')
    # find box dims
    box_min = np.ceil(np.min(molecule.posList)) - 5.0
    box_max = np.ceil(np.max(molecule.posList)) + 5.0
    l_lines.append('%d %d xlo xhi' % (box_min, box_max))
    l_lines.append('%d %d ylo yhi' % (box_min, box_max))
    l_lines.append('%d %d zlo zhi' % (box_min, box_max))
    l_lines.append('')
    l_lines.append('Masses')
    l_lines.append('')
    for i in range(len(molecule.mass)):
        l_lines.append('%d %.5f' % ((i+1), molecule.mass[i]))
    l_lines.append('')
    l_lines.append('Bond Coeffs')
    l_lines.append('')
    for i in range(len(molecule.bondList)):
        l_lines.append('%d %.5f %.5f' % ((i+1), molecule.kb[i], molecule.b0[i]))
    l_lines.append('')
    l_lines.append('Angle Coeffs')
    l_lines.append('')
    for i in range(len(molecule.angleList)):
        l_lines.append('%d %.5f %.5f' % ((i + 1), molecule.kt[i], molecule.t0[i]))
    l_lines.append('')
    l_lines.append('Atoms')
    l_lines.append('')
    for i in range(len(molecule.posList)):
        l_lines.append('%d %d %d %.5f %.5f %.5f' % ((i+1), type_list[i], (i+1), molecule.posList[i,0], molecule.posList[i,1], molecule.posList[i,2]))
    l_lines.append('')
    l_lines.append('Bonds')
    l_lines.append('')
    for i in range(len(molecule.bondList)):
        l_lines.append('%d 1 %d %d' % ((i+1), (molecule.bondList[i,0]+1), (molecule.bondList[i,1]+1)))
    l_lines.append('')
    l_lines.append('Angles')
    l_lines.append('')
    for i in range(len(molecule.angleList)):
        l_lines.append('%d 1 %d %d %d' % ((i+1), (molecule.angleList[i,0]+1), (molecule.angleList[i,1]+1), (molecule.angleList[i,2]+1)))
    l_lines.append('')
    save_file(l_lines, save_dir, fn)
    print('Successfully exported %s to %s' % (fn, save_dir))


def top(mol, ff='amber', fn='cnt.top', save_dir='.'):
    """Creates a .top (topology) type list for use in MD packages
    AMBER99SB or OPLS-AA forcefields can being used"""
    print("Recommended that MD export is done using .gro file only.")
    if ff == 'amber':
        tit = 'AMBER99SB'
        ffloc = './amber99sb.ff/forcefield.itp'
    elif ff == 'oplsaa':
        tit = 'OPLS-AA'
        ffloc = './oplsaa.ff/forcefield.itp'
    else:
        print('Check ff input')
        raise SystemExit
    top_lines = [";", ";     Topology file for %s" % mol.name, ";%s force field" % ff,";"]

    top_lines.extend(["; Include forcefield parameters", '#include "%s"' % ffloc, ""])
    # we call our molecule or residue CNT, encompassing all atoms of the tube/functionalized ends
    top_lines.extend(["[ moleculetype ]", "; Name            nrexcl", "CNT                 3", ""])
    
    # ATOMS
    top_lines.extend(["[ atoms ]", ";   nr    type   resnr  residue   atom    cgnr  charge"])
    for i in range(len(mol.atomtypes)):
        index = i + 1
        if (ff == 'oplsaa') and (mol.atomtypes[i] == 'HC'):
            temp_atomtype = 'HA'
        else:
            temp_atomtype = mol.atomtypes[i]
        a_num = atomtype.inv_atomicSymDict[mol.zList[i]]  # atomic number
        temp_line = "{0:>6}{1:>8}{2:>8}{3:>8}{4:>8}{5:>8}{6:>7.3f}"\
        .format(index, temp_atomtype, 1, 'CNT', a_num, index, 0.000)
        top_lines.append(temp_line)

    # BONDS
    top_lines.extend(["", "[ bonds ]", ";  ai    aj funct           c0           c1"])
    for i in range(len(mol.bondList)):
        funct = 1
        temp_line = "{0:>5}{1:>6}{2:>6}{3:>13}{4:>13}"\
            .format(mol.bondList[i][0]+1, mol.bondList[i][1]+1, funct, "", "")
        top_lines.append(temp_line)

    # PAIRS
    # Let L-J and Coulomb pairs auto generate from the cutoffs

    # ANGLES
    top_lines.extend(["", "[ angles ]", ";  ai    aj    ak funct           c0           c1"])
    for i in range(len(mol.angleList)):
        funct = 1
        temp_line = "{0:>5}{1:>6}{2:>6}{3:>6}{4:>13}{5:>13}"\
            .format(mol.angleList[i][0]+1, mol.angleList[i][1]+1, mol.angleList[i][2]+1, funct, "", "")
        top_lines.append(temp_line)

    # DIHEDRALS
    top_lines.extend(["", "[ dihedrals ]", ";  ai    aj    ak    al funct           c0           c1"])
    for i in range(len(mol.dihList)):
        if ff == 'amber':
            funct = 9
        elif ff == 'oplsaa':
            funct = 3
        temp_line = "{0:>5}{1:>6}{2:>6}{3:>6}{4:>6}{5:>13}{6:>13}"\
            .format(mol.dihList[i][0]+1, mol.dihList[i][1]+1, mol.dihList[i][2]+1, mol.dihList[i][3]+1, funct, "", "")
        top_lines.append(temp_line)

    top_lines.extend(["", "[ system ]", "CNT"])
    top_lines.extend(["", "[ molecules ]", "CNT     1", ""])
    save_file(top_lines, save_dir, fn)
    print('Successfully exported %s to %s' % (fn, save_dir))


def _build_lines(columns, spaces, size, innerColumns):
    listSection = []
    line = ";"
    for column in columns:
        line += ' '*spaces
        line += column
    listSection.append(line)
    for i in range(size):
        line = ' '
        for count, col in enumerate([x[i] for x in innerColumns]):
            entryLength = spaces+len(columns[count])
            line += col.rjust(entryLength, ' ')
        listSection.append(line)
    listSection.append(' ')
    return listSection
    

def save_file(txt_object, save_dir, name):
    f = open(save_dir + "/%s" % name, 'w')
    for i in range(len(txt_object)):
        f.write(txt_object[i] + "\n")
    f.close()

