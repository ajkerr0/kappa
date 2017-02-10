import numpy as np
import re


def pdb(molecule):
    """Creates a .pdb (protein data bank) type list for use with molecular dynamics packages.
    pdb_lines returns list holding every line of .pdb file."""

    atom_lines = []
    conect_lines = []
    inc_index_bondList = molecule.bondList + 1  # sets index to same index as "serial", works
    conect_header_bare = "CONECT"
    for i in range(len(molecule.posList)):
        serial = i + 1
        name = molecule.atomtypes[i]
        altloc = " "
        resname = "CNT"
        chainid = "A"
        resseq = "  %d" % serial
        icode = " "
        x = round(molecule.posList[i][0], 3)
        y = round(molecule.posList[i][1], 3)
        z = round(molecule.posList[i][2], 3)
        occupancy = "1.00"
        tempfactor = "0.00"
        element = molecule.atomtypes[i]
        charge = " "
        atom_header = "ATOM  {0:>5}  {1:<3}{2}{3:>3} {4}{5:>4}{6}   {7:>8}{8:>8}{9:>8}{10:>6}{11:>6}   " \
                      "       {12:>2}{13:>2}".format(serial, name, altloc, resname,
                                                     chainid, resseq, icode, x, y, z,
                                                     occupancy, tempfactor, element, charge)
        atom_lines.append(atom_header)
    for j in range(len(inc_index_bondList)):
        conect_header_temp = conect_header_bare
        for k in range(len(inc_index_bondList[j])):  # builds variable size conect header
            conect_adder = "{0:>5}".format(inc_index_bondList[j][k])
            conect_header_temp += conect_adder
            #conect_a header = conect_header_temp.format(serial, pdb_bonds[i])
        conect_lines.append(conect_header_temp)
    pdb_lines = atom_lines + conect_lines
    pdb_lines.append("TER")
    pdb_lines.append("END")
    return pdb_lines

def gro(molecule, scale=2.0):
    """Creates a .gro file for gromacs, holds atom coordinates and unit cell size
    We assume coordinates are in nm"""
    gro_lines = []
    res_num = 1
    res_name = "CNT"
    elemtypes = []
    # here we check if the atomtype is not a standard element
    for i in range(len(molecule.atomtypes)):
        name_temp = molecule.atomtypes[i]
        regex = re.compile("^[A-Z]{2}$")
        match_bool = re.match(regex, name_temp)
        if match_bool:
            name = name_temp[0]  # strip off second character, ex. CA becomes C
        else:
            name = name_temp
        elemtypes.append(name)
    dist_to_orig = []
    for i in range(len(molecule.posList)):
        temp_dist = np.sqrt(molecule.posList[i][0]**2 + molecule.posList[i][1]**2 + molecule.posList[i][2]**2)
        dist_to_orig.append(temp_dist)
    index_min = np.argmin(dist_to_orig)  # closest pt in object to origin
    # move tube to true origin at 0,0,0
    move_dist = np.abs(molecule.posList[index_min])
    posList_cent = molecule.posList
    posList_cent += move_dist  # now centered at origin
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
    max_length = np.max([length_x, length_y, length_z])
    box_dim = scale * max_length
    print("Box with be %dX the maximum dimension of the object.\nUsing a %dX%dX%d box." %
          (scale, box_dim, box_dim, box_dim))
    new_move_dist = box_dim/2.0
    # measure dist to new box origin in move in every direction
    posDist_new = posList_cent
    posDist_new += new_move_dist
    # now tube is centered in quadrant 0 box
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
            .format(res_num, res_name, elemtypes[i], _index,
                    posDist_new[i][0], posDist_new[i][1], posDist_new[i][2])
        gro_lines.append(temp_line)
    box_line = "{0:>8.3f}{1:>8.3f}{2:>8.3f}".format(box_dim, box_dim, box_dim)
    gro_lines.append(box_line)
    return gro_lines


def top(mol):
    """Creates a .top (topology) type list for use in MD packages"""
    
    top_lines = [";", ";     Topology file for %s" % mol.name, ";"]
    
    #defaults line
    #used to establish default parameters (for when they aren't specified)????
    #copy and pasting from example file
    top_lines.extend(["[ defaults ]",
                      "; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ",
                      "  1             1               no              1.0     1.0",
                      ""])
    
    #inclue forcefield field file
    top_lines.extend(["; The force field files to be included",
                      '#include "%s.itp"' % mol.ff.name,
                      ""])
                      
    #ignoring moleculetype
    #maybe add a molecule type attribute to our molecules?
    
    #atoms
    top_lines.append("[ atoms ]")
#    top_lines.extend(_build_lines(["nr", "type", "cgnr", "charge"],
#                                  5,
#                                  len(mol),
#                                  map(str, [range(len(mol)), mol.atomtypes, 
#                                       np.ones(len(mol), dtype=int), np.zeros(len(mol), dtype=float)])))
    columns = ["nr", "type", "cgnr", "charge"]
    spaces = 5
    line = ";"
    for column in columns:
        line += ' '*spaces
        line += column
    top_lines.append(line)
    for i in range(len(mol)):
        atomcols = map(str, [i, mol.atomtypes[i], 1, 0.000])
        line = ' '
        for count, col in enumerate(atomcols):
            entryLength = spaces+len(columns[count])
            line += col.rjust(entryLength, ' ')
        top_lines.append(line)
    top_lines.append(" ")
        
    #bonds
    top_lines.append("[ bonds ]")
    columns = ["ai", "aj", "funct", "c0", "c1"]
    spaces = 12
    line = ";"
    for column in columns:
        line += ' '*spaces
        line += column
    top_lines.append(line)
    for m, bond in enumerate(mol.bondList):
        bondcols = map(str, [bond[0], bond[1], 1, mol.kb[m], mol.b0[m]])
        line = " "
        for count, col in enumerate(bondcols):
            entryLength = spaces+len(columns[count])
            line += col.rjust(entryLength, ' ')
        top_lines.append(line)
            
    return top_lines
    
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

