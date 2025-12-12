import argparse, sys, os
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv

# Map element type to (number of nodes, VTK cell type)
VTK_CELL_TYPES = {
    "tri3":  (3, 5),   
    "tri6":  (6, 22),  
    "quad4": (4, 9),   
    "quad8": (8, 23),
}
# Map element type to (number of nodes)
elem_nodes = {
    "tri3": 3,
    "tri6": 6,
    "quad4": 4,
    "quad8": 8,
    "quad9": 9
}
def pars_feb(input_file, input_type):

    pts = []
    elems = [] 
    all_surface = {}
    ED_fields = {}

    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        

        for nodes_group in root.iter('Nodes'):
            for node in nodes_group.findall('node'):
                node_id_str = node.attrib['id'] 
                coords_str = node.text.strip()  
                try:
                    x, y, z = map(float, coords_str.split(','))
                    field = (x, y, z) 
                    pts.append(field) # Use .add() on the set
                except ValueError as e:
                    print(f"Warning: Could not parse coordinates for node ID {node_id_str}: {coords_str} - {e}", file=sys.stderr)

        for elems_group in root.iter('Elements'): 
            for elem in elems_group.findall('elem'): 
                elem_id_str = elem.attrib['id'] 
                elem_str = elem.text.strip()  
                try:
                    integer_list = list(map(int, elem_str.split(',')))
                    elems.append(integer_list) 
                except ValueError as e:
                    print(f"Warning: Could not parse Elements for elems ID {elem_id_str}: {elem_str} - {e}", file=sys.stderr)
        
        for surface_group in root.iter('Surface'):
            surface_name = surface_group.attrib.get('name')
            surface_elems = [] 
            # for child in surface_group:
            #     print(f"  - {child.tag}") 
            for elem in surface_group.findall(input_type): 
                elem_id_str = elem.attrib['id'] 
                elem_str = elem.text.strip()  
                try:
                    integer_list = list(map(int, elem_str.split(',')))
                    surface_elems.append(integer_list)
                except ValueError as e:
                    print(f"Warning: Could not parse Elements for elems ID {elem_id_str}: {elem_str} - {e}", file=sys.stderr)
            all_surface[surface_name] = surface_elems 

        for ElementData_group in root.iter('ElementData'):
            ElementData_name = ElementData_group.attrib.get('name','shell_thickness')
            ElementData_elems = [] 
            # for child in ElementData_group:
            #     print(f"  - {child.tag}") 
            for elem in ElementData_group.findall('e'): 
                elem_id_str = elem.attrib['lid'] 
                elem_str = elem.text.strip()  
                try:
                    integer_list = list(map(float, elem_str.split(',')))
                    ElementData_elems.append(integer_list)
                except ValueError as e:
                    print(f"Warning: Could not parse Elements for elems ID {elem_id_str}: {elem_str} - {e}", file=sys.stderr)
            ED_fields[ElementData_name] = ElementData_elems 

    except FileNotFoundError:
        print(f"ERROR: Can not find the file {input_file}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError:
        print(f"ERROR: Could not parse {input_file} as a valid XML file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
        
    return {
        'pts': pts,
        'elems': elems,
        'thickness': ED_fields.get('shell_thickness'),
        'surface': all_surface,
        'ED': {k: v for k, v in ED_fields.items() if k != 'shell_thickness'}
    }
def write_vtk(points, elems, elem_type, point_data=None, cell_data=None, out_vtk="output.vtk"):
    """
    Write an unstructured mesh of various types to VTK with optional nodal and cell data.

    Parameters
    ----------
    points : (N, 3) array
        Coordinates of the nodes.
    elems : (M, n_nodes) array
        Connectivity of elements.
    elem_type : str
        Element type: "tri3", "tri6", "quad4", "quad8".
    point_data : dict[str, np.ndarray], optional
        Nodal fields; each array must have length N.
    cell_data : dict[str, np.ndarray], optional
        Element fields; each array must have length M.
    out_vtk : str
        Output file path.
    """
    points = np.asarray(points)
    elems = np.asarray(elems, dtype=int)

    if elem_type not in VTK_CELL_TYPES:
        raise ValueError(f"Unknown element type: {elem_type}")

    n_nodes, vtk_cell_type = VTK_CELL_TYPES[elem_type]
    n_elems = elems.shape[0]

    if elems.shape[1] != n_nodes:
        raise ValueError(f"Expected {n_nodes} nodes/element for {elem_type}, got {elems.shape[1]}")

    # Build VTK cell array
    cell_array = np.hstack([np.full((n_elems, 1), n_nodes, dtype=np.int64), elems]).flatten()
    cell_types = np.full(n_elems, vtk_cell_type, dtype=np.uint8)

    # Create the unstructured grid
    grid = pv.UnstructuredGrid(cell_array, cell_types, points)

    # Add point fields
    if point_data:
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            if len(arr) != len(points):
                raise ValueError(f"Nodal field '{name}' length {len(arr)} != number of points {len(points)}")
            grid.point_data[name] = arr

    # Add cell fields
    if cell_data:
        for name, arr in cell_data.items():
            arr = np.asarray(arr)
            if len(arr) != n_elems:
                raise ValueError(f"Cell field '{name}' length {len(arr)} != number of elements {n_elems}")
            grid.cell_data[name] = arr

    # Write file
    grid.save(out_vtk, binary=False)
    print(f"* Writing information from an input file into a VTK format :{out_vtk}!")
def tri3_to_tri6(points, elems, point_data=None, cell_data=None):
    """
    Convert a tri3 mesh to tri6 by inserting mid-edge nodes.
    points: (N,3) Standard Python list
    elems: (M,3) 1-based a standard Python list
    cell_data: Dictionary of Python lists
    cell_data: Dictionary of Python lists
    """
    points = np.asarray(points, dtype=float)
    elems = np.asarray(elems, dtype=int)
    npts = points.shape[0]
    ntri = elems.shape[0]
    new_points = points.tolist()
    new_elems = np.zeros((ntri, 6), dtype=int)

    # copy cell data
    new_cell_data = {}
    if cell_data is not None:
        for name, arr in cell_data.items():
            arr = list(arr)
            new_cell_data[name] = arr[:]

    # nodal data
    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            arr = list(arr)
            new_point_data[name] = arr[:]  


    mid_map = {}
    def get_mid(a, b):
        # convert to zero-based
        ia = a - 1
        ib = b - 1
        key = (min(a, b), max(a, b))

        if key in mid_map:
            return mid_map[key]

        # compute new midpoint coordinates
        pa = points[ia]
        pb = points[ib]
        mid = 0.5 * (pa + pb)

        idx = len(new_points) + 1  # 1-based 
        new_points.append(mid.tolist())

        # interpolate nodal scalar fields
        if point_data is not None:
            for name, arr in new_point_data.items():
                va = arr[ia]
                vb = arr[ib]
                arr.append(0.5 * (va + vb))

        mid_map[key] = idx
        return idx

    # build new tri6 connectivity
    for i in range(ntri):
        a, b, c = elems[i]

        m01 = get_mid(a, b)
        m12 = get_mid(b, c)
        m20 = get_mid(c, a)

        new_elems[i] = [a, b, c, m01, m12, m20]

    print(f"* Successfully Converted from tri3 to tri6!")
    return np.asarray(new_points), new_elems, new_point_data, new_cell_data
def tri6_to_tri3(points, elems, point_data=None, cell_data=None):
    """
    Convert a tri6 mesh to tri3 by:
    - extracting vertex nodes only
    - removing mid-edge nodes
    - remapping connectivity
    - trimming nodal data to only used nodes
    """

    points = np.asarray(points, float)
    elems = np.asarray(elems, int)   
    ntri = elems.shape[0]
    tri3 = elems[:, :3].copy()    
    used_nodes = np.unique(tri3)          # 1-based node IDs
    used_nodes_zero = used_nodes - 1      # convert to 0-based indexing

    # Build map: old_index → new_index (both 1-based)
    remap = {old: new for new, old in enumerate(used_nodes, start=1)}
    new_points = points[used_nodes_zero, :].copy()

    # Remap tri3 connectivity to new node indices
    new_elems = np.zeros_like(tri3)
    for i in range(ntri):
        a, b, c = tri3[i]
        new_elems[i, 0] = remap[a]
        new_elems[i, 1] = remap[b]
        new_elems[i, 2] = remap[c]

    # Trim nodal data (remove mid-edge values)
    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            new_point_data[name] = arr[used_nodes_zero].copy()

    # Copy cell_data unchanged (same number of triangles)
    new_cell_data = {}
    if cell_data is not None:
        for name, arr in cell_data.items():
            new_cell_data[name] = np.asarray(arr).copy()

    print("* Successfully converted tri6 → tri3 (mid-edge nodes removed).")
    return new_points, new_elems, new_point_data, new_cell_data
def tri3_to_quad4(points, elems, point_data=None, cell_data=None):
    """
    Convert each tri3 element into 3 quad4 elements.
    points: (N,3) array-like, 1-based connectivity
    elems: (M,3) array-like (1-based)
    """
    points = np.asarray(points, dtype=float)
    elems = np.asarray(elems, dtype=int)

    new_points = points.tolist()   
    new_elems = []                 

    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            new_point_data[name] = np.asarray(arr).tolist()

    new_cell_data = {}
    if cell_data is not None:
        for name in cell_data.keys():
            new_cell_data[name] = []

    mid_map = {}
    def get_mid(a, b):
        """Return the 1-based index of the midpoint of edge (a,b)."""
        key = (min(a, b), max(a, b))
        if key in mid_map:
            return mid_map[key]

        ia = a - 1
        ib = b - 1
        pa = points[ia]
        pb = points[ib]

        mid = 0.5 * (pa + pb)

        idx = len(new_points) + 1  
        new_points.append(mid.tolist())

        if point_data is not None:
            for name, arr in new_point_data.items():
                va = arr[ia]
                vb = arr[ib]
                arr.append(0.5 * (va + vb))

        mid_map[key] = idx
        return idx

    for i in range(elems.shape[0]):
        a, b, c = elems[i]
        ia, ib, ic = a - 1, b - 1, c - 1

        # Mid-edge nodes
        ab = get_mid(a, b)
        bc = get_mid(b, c)
        ca = get_mid(c, a)

        # Centroid
        centroid_coord = (points[ia] + points[ib] + points[ic]) / 3.0
        centroid_idx = len(new_points) + 1
        new_points.append(centroid_coord.tolist())

        if point_data is not None:
            for name, arr in new_point_data.items():
                va = arr[ia]
                vb = arr[ib]
                vc = arr[ic]
                arr.append((va + vb + vc) / 3.0)

        # Build the 3 quads, 1-based indexing
        q1 = [a, ab, centroid_idx, ca]
        q2 = [b, bc, centroid_idx, ab]
        q3 = [c, ca, centroid_idx, bc]

        new_elems.append(q1)
        new_elems.append(q2)
        new_elems.append(q3)

        # Duplicate cell data for the 3 generated quads
        if cell_data is not None:
            for name, arr in new_cell_data.items():
                v = cell_data[name][i]
                arr.extend([v, v, v])

    print(f"* Successfully Converted from tri3 to quad4!")
    return np.asarray(new_points), new_elems, new_point_data, new_cell_data
def quad4_to_quad8(points, elems, point_data=None, cell_data=None):
    """
    Convert a quad4 mesh to quad8 by inserting mid-edge nodes.
    Nodes are 1-based in elems.
    """

    points = np.asarray(points, float)
    elems = np.asarray(elems, int)
    nelem = elems.shape[0]
    new_points = points.tolist()

    # copy cell data
    new_cell_data = {}
    if cell_data is not None:
        for name, arr in cell_data.items():
            new_cell_data[name] = list(arr)

    # copy nodal data
    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            new_point_data[name] = list(arr)

    # dictionary to avoid duplicate midpoints
    mid_map = {}
    def get_mid(a, b):
        """Return index of midpoint node (1-based)."""
        key = (min(a, b), max(a, b))
        if key in mid_map:
            return mid_map[key]

        ia = a - 1
        ib = b - 1

        pa = points[ia]
        pb = points[ib]
        mid = 0.5 * (pa + pb)

        # new index (1-based)
        idx = len(new_points) + 1
        new_points.append(mid.tolist())

        # interpolate nodal data
        if point_data is not None:
            for name, arr in new_point_data.items():
                va = arr[ia]
                vb = arr[ib]
                arr.append(0.5 * (va + vb))

        mid_map[key] = idx
        return idx

    # Build quad8 connectivity
    new_elems = []
    for i in range(nelem):
        a, b, c, d = elems[i]

        m01 = get_mid(a, b)
        m12 = get_mid(b, c)
        m23 = get_mid(c, d)
        m30 = get_mid(d, a)

        new_elems.append([a, b, c, d, m01, m12, m23, m30])

    print("* Successfully converted quad4 → quad8!")
    return np.asarray(new_points), np.asarray(new_elems, int), new_point_data, new_cell_data
def quad8_to_quad9(points, elems, point_data=None, cell_data=None):
    """
    Convert a quad8 mesh to quad9 by inserting centroid nodes.
    elems must be 1-based indexing.
    """

    points = np.asarray(points, float)
    elems = np.asarray(elems, int)
    new_points = points.tolist()

    # Copy cell data
    new_cell_data = {}
    if cell_data is not None:
        for name, arr in cell_data.items():
            new_cell_data[name] = list(arr)

    # Copy point data 
    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            new_point_data[name] = list(arr)

    new_elems = []

    for elem in elems:
        a, b, c, d, m1, m2, m3, m4 = elem
        ia, ib, ic, id = a - 1, b - 1, c - 1, d - 1

        # Compute centroid
        centroid = (points[ia] + points[ib] + points[ic] + points[id]) / 4.0
        centroid_idx = len(new_points) + 1  # 1-based index
        
        new_elems.append([a, b, c, d, m1, m2, m3, m4, centroid_idx])
        new_points.append(centroid.tolist())

        # Interpolate point data at centroid
        if point_data is not None:
            for name, arr in new_point_data.items():
                arr.append((arr[ia] + arr[ib] + arr[ic] + arr[id]) / 4.0)

    print("* Successfully converted quad8 → quad9!")
    return np.asarray(new_points), np.asarray(new_elems, int), new_point_data, new_cell_data
def quad4_to_tri3(points, quads, point_data=None, cell_data=None):
    """
    Convert quad4 mesh to tri3 mesh.
    points: (N,3) array-like
    quads: (M,4) array-like (1-based connectivity)
    """
    points = np.asarray(points, float)
    quads = np.asarray(quads, int)

    # Copy nodal data directly
    new_points = points.tolist()
    new_point_data = {}
    if point_data is not None:
        for name, arr in point_data.items():
            new_point_data[name] = list(arr)

    # Initialize cell data arrays (each quad → 2 triangles)
    new_elems = []
    new_cell_data = {}
    if cell_data is not None:
        for name in cell_data:
            new_cell_data[name] = []

    for i in range(quads.shape[0]):

        a, b, c, d = quads[i]  # 1-based node IDs

        # Build 2 triangles
        tri1 = [a, b, c]
        tri2 = [a, c, d]

        new_elems.append(tri1)
        new_elems.append(tri2)

        # Duplicate cell data for each quad → 2 tris
        if cell_data is not None:
            for name in new_cell_data:
                v = cell_data[name][i]
                new_cell_data[name].extend([v, v])

    print("* Successfully Converted from quad4 to 2 tri3!")
    return np.asarray(new_points), np.asarray(new_elems, int), new_point_data, new_cell_data

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and not elem.tail or not elem.tail.strip():
            elem.tail = i
def write_feb_file(input_filepath, output_filepath,mesh_type, new_coordinates, new_connectivity, new_surface, new_ED):
    try:        
        tree = ET.parse(input_filepath)
        root = tree.getroot()

        nodes_block = root.find('.//Nodes') 
        element_block = root.find('.//Elements')
        surface_blocks = root.findall(".//Surface")
        ED_blocks = root.findall(".//ElementData")

        if nodes_block is None:
            print("Error: Could not find the <Nodes> tag.")
            exit(1)
        if element_block is None:
            print("Error: Could not find the <Elements> tag.")
            exit(1)
        if surface_blocks is None:
            print("Error: Could not find the <Surface> tag.")
            exit(1)
        if ED_blocks is None:
            print("Error: Could not find the <ElementData> tag.")
            exit(1)
        
        # --- Cleaning and replacing Nodes ---
        for child in list(nodes_block): 
            nodes_block.remove(child)
        node_id = 1
        for coords in new_coordinates:
            coord_text = ', '.join(map(str, coords))
            new_node = ET.SubElement(nodes_block, 'node', 
                                     attrib={'id': str(node_id)})
            new_node.text = coord_text
            node_id += 1 

        print("   -> All new nodes added successfully.")
        
        # --- Cleaning and replacing Elements ---
        element_block.set('type', mesh_type)
        for child in list(element_block): 
            element_block.remove(child)
        elem_id = 1
        for pid in new_connectivity:
            pid = [int(value) for value in pid]
            coord_text = ', '.join(map(str, pid))
            new_node = ET.SubElement(element_block, 'elem', 
                                     attrib={'id': str(elem_id)})
            new_node.text = coord_text
            elem_id += 1 
        print("   -> All new Elements added successfully.")
        
        # --- Cleaning and replacing Surfaces ---
        for surface_block, (surface_name, field) in zip(surface_blocks, new_surface.items()):
            # clear old children
            for child in list(surface_block):
                surface_block.remove(child)

            # update name
            surface_block.set('name', surface_name)

            if len(field) != len(new_connectivity):
                print(f"ERROR: length of {surface_name} ({len(field)}) does not match number of elements ({len(new_connectivity)})!")
                exit(1)

            surf_id = 1
            for elem_id, value in enumerate(field):
                if value == 1:
                    pid = [int(n) for n in new_connectivity[elem_id]]
                    coord_text = ', '.join(map(str, pid))

                    new_node = ET.SubElement(surface_block, mesh_type, attrib={'id': str(surf_id)})
                    new_node.text = coord_text

                    surf_id += 1

        print("   -> All new surfaces added successfully.")

        
       
        # --- Cleaning and replacing ElementData ---
        for ED_block, (ED_name, field) in zip(ED_blocks, new_ED.items()):
            # clear old children
            for child in list(ED_block):
                ED_block.remove(child)
            # clear old children
            if ED_name == "shell thickness":
                if len(field) != len(new_coordinates):
                    print("ERROR: length of the thickness field is not match with number of points!")
                    exit(1)
                # update name
                ED_block.set('type', ED_name)
                if "name" in ED_block.attrib:
                    del ED_block.attrib["name"]
                for ele, elem in enumerate(new_connectivity):
                    thickness_strings = [str(float(field[int(pid)-1])) for pid in elem]
                    coord_text = ', '.join(thickness_strings)
                    new_node = ET.SubElement(ED_block, 'e', 
                                            attrib={'lid': str(ele+1)})
                    new_node.text = coord_text
                print("   -> New thickness Element Data added successfully.")
            else:
                if len(field) != len(new_connectivity):
                    print(f"ERROR: length of the {ED_name}({len(field)}) field is not match with number of element!")
                    exit(1)
                # update name
                ED_block.set('name', ED_name)
                if "type" in ED_block.attrib:
                    del ED_block.attrib["type"]
                for ele in range(len(new_connectivity)):
                    new_node = ET.SubElement(ED_block, 'e', 
                                            attrib={'lid': str(ele+1)})
                    new_node.text = str(field[int(ele)-1][0])

                print("   -> All new Element Data fields added successfully.")
        
        
        # Sanitizied the output (single-line output): 
        indent(root)

        # Save the modified XML to a new file
        tree.write(output_filepath, encoding='utf-8', xml_declaration=True)
        print(f"file saved to: {os.path.basename(output_filepath)}")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

ap = argparse.ArgumentParser(description="This script reads an input FEB file and converts it to a new mesh type!")
ap.add_argument('input_file', help='Input FEB file')
ap.add_argument('input_type', help='The mesh type for the input FEB file')
ap.add_argument('output_type', help='The mesh type for the output FEB file')
ap.add_argument('-f', '--finner', dest='finner', action='store_true', help='Refinnering input mesh')
ap.add_argument('-n', '--name', dest='output_filename', type=str, help='Name of output file')

arg = ap.parse_args()

input_file = arg.input_file
output_file = 'converted.feb' if arg.output_filename is None else arg.output_filename
input_type = arg.input_type
output_type = arg.output_type

# check finner option
finner_mesh = False
if arg.finner:
    if input_type not in ['tri3', 'tri6', 'quad4']:
        raise ValueError(
            f"Input mesh type must be 'tri3', 'tri6', or 'quad4', but got '{input_type}'!"
        )
    finner_mesh = True

# check input and output
if input_type == output_type and finner_mesh == False:
    raise ValueError(f"Both mesh types are same!")

# check input_type 
n_nodes = elem_nodes.get(input_type)
if n_nodes is None:
    raise ValueError(f"Unknown element type: {input_type}")    
input_elem_dtype = np.dtype([(f"n{i}", int) for i in range(n_nodes)])

# check output_type 
n_nodes = elem_nodes.get(output_type)
if n_nodes is None:
    raise ValueError(f"Unknown element type: {output_type}")    

# Parse the input feb file  *.feb
input_mesh = pars_feb(input_file,input_type)
print(f"* Number of points: {len(input_mesh['pts'])}")
print(f"* Number of elements: {len(input_mesh['elems'])}")
print(f"* Number of thickness: {len(input_mesh['thickness'])}")
for surface_name, elements_set in input_mesh['surface'].items():
    print(f"* Surface Name: '{surface_name}'")
    print(f"\t- Number of elements in this surface: {len(elements_set)}")
for ElementData_name, elements_set in input_mesh['ED'].items():
    print(f"* Element Data Name: '{ElementData_name}'")
    print(f"\t- Number of elements in this Element Data: {len(elements_set)}")

# converted pointal thickness 
npts = len(input_mesh['pts'])
pts_thickness = [0.0] * npts
for ele in range(len(input_mesh['elems'])):
    nodes = input_mesh['elems'][ele]       
    thick = input_mesh['thickness'][ele]   
    for i, node_id in enumerate(nodes):
        pts_thickness[node_id - 1] = thick[i]
print(f"* Number of nodal thickness: {len(pts_thickness)}")

# Converted surface to mask field 
all_elems_list = list(input_mesh['elems'])  
all_elems_array = np.array(all_elems_list, dtype=int)
elems_struct = all_elems_array.view(input_elem_dtype)
mask_surface = {}
for surface_name, elements_set in input_mesh['surface'].items():
    surface_elems = np.array(list(elements_set), dtype=int).view(input_elem_dtype)
    mask = np.isin(elems_struct, surface_elems)
    mask_surface[surface_name] = mask.astype(int)

# save cell_data and nodal_data in vtk file 
corrected_mesh=[]
for elem in input_mesh['elems']:
    array = []
    for ele in elem:
        array.append(int(ele)-1)
    corrected_mesh.append(array)
point_data = {
    'shell thickness': pts_thickness
}
cell_data = {}
cell_data.update(mask_surface)
cell_data.update(input_mesh['ED'])
write_vtk(input_mesh['pts'], corrected_mesh,input_type,cell_data=cell_data, point_data=point_data, out_vtk=f"{os.path.splitext(os.path.basename(input_file))[0]}_{input_type}.vtk")

# Convert Mesh
if not finner_mesh:
    if input_type == "tri3" and output_type == "tri6":
        new_pts, new_conn, new_point_data, new_cell_data = tri3_to_tri6(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
    if input_type == "tri6" and output_type == "tri3":
        new_pts, new_conn, new_point_data, new_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )    
    if input_type == "tri3" and output_type == "quad4":
        new_pts, new_conn, new_point_data, new_cell_data = tri3_to_quad4(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
    if input_type == "tri6" and output_type == "quad4":
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        new_pts, new_conn, new_point_data, new_cell_data = tri3_to_quad4(tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data)    
    if input_type == "quad4" and output_type == "quad8":
        new_pts, new_conn, new_point_data, new_cell_data = quad4_to_quad8(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
    if input_type == "tri3" and output_type == "quad8":
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        new_pts, new_conn, new_point_data, new_cell_data = quad4_to_quad8(quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data) 
    if input_type == "quad8" and output_type == "quad9":
        new_pts, new_conn, new_point_data, new_cell_data = quad8_to_quad9(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
    if input_type == "tri3" and output_type == "quad9":
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data = quad4_to_quad8(quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = quad8_to_quad9(quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data)
    if input_type == "tri6" and output_type == "quad8":
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = quad4_to_quad8(quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
    if input_type == "tri6" and output_type == "quad9":
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data)
        quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data = quad4_to_quad8(quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = quad8_to_quad9(quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data)  
    if input_type == "tri6" and output_type == "quad9":
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data)
        quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data = quad4_to_quad8(quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = quad8_to_quad9(quad8_pts, quad8_conn, quad8_point_data, quad8_cell_data)   
else:
    if input_type == "tri3" and output_type == "tri3":
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        new_pts, new_conn, new_point_data, new_cell_data = quad4_to_tri3( quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
    if input_type == "tri3" and output_type == "tri6":
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = quad4_to_tri3( quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = tri3_to_tri6( tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data )
    if input_type == "tri6" and output_type == "tri6":
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = tri6_to_tri3(
            input_mesh['pts'],
            input_mesh['elems'],
            point_data=point_data,
            cell_data=cell_data
        )
        quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data = tri3_to_quad4(tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data)
        tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data = quad4_to_tri3( quad4_pts, quad4_conn, quad4_point_data, quad4_cell_data)
        new_pts, new_conn, new_point_data, new_cell_data = tri3_to_tri6( tri3_pts, tri3_conn, tri3_point_data, tri3_cell_data )
# Convert surface masks and ED 
new_mask_surface = {}
for surf_name, old_field in mask_surface.items():
    if surf_name in new_cell_data:
        new_mask_surface[surf_name] = np.asarray(new_cell_data[surf_name])
    else:
        print(f"WARNING: surface '{surf_name}' not found in new_cell_data")


new_ED = {}
for ed_name, old_field in input_mesh['ED'].items():
    if ed_name in new_cell_data:
        new_ED[ed_name] = np.asarray(new_cell_data[ed_name])
    else:
        print(f"WARNING: ED '{ed_name}' not found in new_cell_data")
updated_ED = new_ED.copy()
if 'shell thickness' in new_point_data:
    updated_ED['shell thickness'] = np.asarray(new_point_data['shell thickness'])

# Write the converted feb file 
write_feb_file(input_file, output_file,output_type,new_pts, new_conn,new_mask_surface,updated_ED)



