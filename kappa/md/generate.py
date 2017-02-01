import numpy as np

from ..antechamber.atomtype import atomtype


def pdb(molecule):
    """Creates a .pdb (protein data bank) type list for use with molecular dynamics packages.
    pdb_lines returns list holding every line of .pdb file."""

    atom_lines = []
    conect_lines = []
    inc_index_bondList = molecule.bondList + 1  # sets index to same index as "serial", works
    pdb_bonds = []
    conect_header_bare = "CONECT"
    for i in range(len(molecule.posList)):
        serial = i + 1
        name = atomtype.inv_atomicSymDict[molecule.zList[i]]
        altloc = " "
        #resname = "UNK"
        resname = "%s  " % name
        chainid = " "
        resseq = 1
        icode = " "
        x = round(molecule.posList[i][0], 3)
        y = round(molecule.posList[i][1], 3)
        z = round(molecule.posList[i][2], 3)
        occupancy = "1.00"
        tempfactor = "0.00"
        element = " %s" % name
        charge = " "
        atom_header = "ATOM  {0:>5}  {1:<3}{2}{3:<3} {4}{5:>4}{6}   {7:>8}{8:>8}{9:>8}{10:>6}{11:>6}   " \
                      "       {12:>2}{13:>2}".format(serial, name, altloc, resname, chainid, resseq, icode, x, y, z,
                                              occupancy, tempfactor, element, charge)
        atom_lines.append(atom_header)
    for j in range(len(inc_index_bondList)):
        conect_header_temp = conect_header_bare
        for k in range(len(inc_index_bondList[j])):  # builds variable size conect header
            conect_adder = "{0:>5}".format(inc_index_bondList[j][k])
            conect_header_temp += conect_adder
            #conect_header = conect_header_temp.format(serial, pdb_bonds[i])
        conect_lines.append(conect_header_temp)
    pdb_lines = atom_lines + conect_lines
    pdb_lines.append("TER")
    pdb_lines.append("END")
    return pdb_lines
    
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

