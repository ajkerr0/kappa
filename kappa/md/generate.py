import elements as ELEMENTS


def pdb(molecule):
    # Creates a .pdb (protein data bank) type list for use with molecular dynamics packages
    # pdb_lines returns list holding every line of .pdb file
    pdb_lines = []
    atom_lines = []
    conect_lines = []
    inc_index_bondList = molecule.bondList + 1  # sets index to same index as "serial"
    pdb_bonds = []
    conect_header_bare = "CONECT {0:>5}"
    for i in range(len(molecule.posList)):
        serial = i + 1
        name = ELEMENTS[molecule.zList[i]].symbol
        altloc = " "
        resname = "UNK"
        chainid = " "
        resseq = 1
        icode = " "
        x = round(molecule.posList[i][0], 3)
        y = round(molecule.posList[i][1], 3)
        z = round(molecule.posList[i][2], 3)
        occupancy = 1.00
        tempfactor = 0.00
        element = ELEMENTS[molecule.zList[i]].symbol
        charge = "  "
        pdb_bonds.append([])
        for j in range(len(inc_index_bondList)):
            counteachbond = 0
            if molecule.bondList[j][0] == i:  # checks first index
                counteachbond += 1
                pdb_bonds[-1].append(molecule.bondList[j][1] + 1)  # generate correct bond format
                if counteachbond > 4:
                    print("Too many bonds for standard pdb specs.")
                    raise SystemExit
            counteachbond = 0
            if molecule.bondList[j][1] == i:  # checks second index
                counteachbond += 1
                pdb_bonds[-1].append(molecule.bondList[j][0] + 1)  # generate correct bond format
                if counteachbond > 4:
                    print("Too many bonds for standard pdb specs.")
                    raise SystemExit
                    # behavior above double counts CONECT bonds, as per PDB specs
        atom_header = "ATOM  {0:>5} {1:>4}{2}{3:>3} {4}{5:>4}{6}   {7:>7}{8:>7}{9:>7}{10:>6}{11:>6}          " \
                      "{12:>2}{13:>2}".format(serial, name, altloc, resname, chainid, resseq, icode, x, y, z,
                                              occupancy, tempfactor, element, charge)
        conect_header_temp = conect_header_bare
        for k in range(len(pdb_bonds[i])):  # builds variable size conect header
            conect_adder = "{1[%d]:>5}" % k
            conect_header_temp += conect_adder
        conect_header = conect_header_temp.format(serial, pdb_bonds[i])
        atom_lines.append(atom_header)
        conect_lines.append(conect_header)
    pdb_lines = atom_lines + conect_lines
    pdb_lines.append("TER")
    pdb_lines.append("END")
    return pdb_lines


def save_file(txt_object, save_dir, name):
    f = open(save_dir + "/%s.pdb" % name, 'w')
    for i in range(len(txt_object)):
        f.write(txt_object[i] + "\n")
    f.close()

