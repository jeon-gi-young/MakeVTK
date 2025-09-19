import os
import configparser
from datetime import datetime
import numpy as np
import pandas as pd
from pyFAST.input_output.fast_output_file import FASTOutputFile

def read_config(ini_path):
    if not os.path.exists(ini_path):
        print(f"Warning: Configuration file '{ini_path}' not found.")
        raise FileNotFoundError(f"Configuration file '{ini_path}' does not exist.")
    config = configparser.ConfigParser()
    config.read(ini_path, encoding='utf-8')
    input_cfg = config['input']
    output_cfg = config['output']
    return input_cfg, output_cfg

def setup_output_folder(path_out):
    if not path_out.endswith(os.sep):
        path_out += os.sep
    if not os.path.exists(path_out):
        print('Output folder is generated.')
        os.makedirs(path_out)
    return path_out

def parse_nastran_bulk(filename):
    # First pass: count GRID, CTRIA3, CQUAD4
    ngr, nt3, nq4 = 0, 0, 0
    with open(filename, 'r') as f:
        for line in f:
            if len(line) > 8:
                cardname = line[:8]
                if cardname == 'GRID    ':
                    ngr += 1
                elif cardname == 'CTRIA3  ':
                    nt3 += 1
                elif cardname == 'CQUAD4  ':
                    nq4 += 1
    nodes = np.full((ngr, 1), np.nan)
    elems = np.full((nt3 + nq4, 1), np.nan)
    points = np.full((ngr, 3), np.nan)
    poly3 = np.full((nt3, 3), np.nan)
    poly4 = np.full((nq4, 4), np.nan)
    # Second pass: extract data
    npt, npl3, npl4 = 0, 0, 0
    with open(filename, 'r') as f:
        for line in f:
            if len(line) > 8:
                cardname = line[:8]
                if cardname == 'GRID    ':
                    npt += 1
                    nodes[npt-1, 0] = int(line[8:16])
                    points[npt-1, 0] = float(line[24:32])
                    points[npt-1, 1] = float(line[32:40])
                    points[npt-1, 2] = float(line[40:48])
                elif cardname == 'CTRIA3  ':
                    npl3 += 1
                    elems[npl3-1, 0] = int(line[8:16])
                    poly3[npl3-1, 0] = int(line[24:32])
                    poly3[npl3-1, 1] = int(line[32:40])
                    poly3[npl3-1, 2] = int(line[40:48])
                elif cardname == 'CQUAD4  ':
                    npl4 += 1
                    elems[npl4-1, 0] = int(line[8:16])
                    poly4[npl4-1, 0] = int(line[24:32])
                    poly4[npl4-1, 1] = int(line[32:40])
                    poly4[npl4-1, 2] = int(line[40:48])
                    poly4[npl4-1, 3] = int(line[48:56])
    points /= 1e3  # mm to m
    return nodes, points, poly3, poly4

def extract_tower_base_height(ed_file):
    TowerBsHt = None
    with open(ed_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2 and parts[1] == 'TowerBsHt':
                TowerBsHt = float(parts[0])
                break
    if TowerBsHt is None:
        raise ValueError('TowerBsHt is not defined in ElastoDyn input file.')
    return [0, 0, TowerBsHt]

def extract_ptfm_ref_coords(hd_file):
    ptfm_ref = [None, None, None]
    found = 0
    with open(hd_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2:
                if parts[1] == 'PtfmRefxt':
                    ptfm_ref[0] = float(parts[0])
                    found += 1
                elif parts[1] == 'PtfmRefyt':
                    ptfm_ref[1] = float(parts[0])
                    found += 1
                elif parts[1] == 'PtfmRefzt':
                    ptfm_ref[2] = float(parts[0])
                    found += 1
            if found == 3:
                break
    if found != 3:
        raise ValueError('Check PtfmRefxt, PtfmRefyt, PtfmRefzt in HydroDyn input file.')
    return ptfm_ref

def extract_fairlead_coords(md_file):
    fairleads = []
    with open(md_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2 and parts[1] == 'Vessel':
                fairleads.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not fairleads:
        raise ValueError("There is no 'Vessel' points in MoorDyn input file.")
    return fairleads

def orphan_node_deletion(nodes, points, poly3, poly4):
    used_nodes = np.unique(np.concatenate([poly3.flatten(), poly4.flatten()]))
    is_orphan = ~np.isin(nodes.flatten(), used_nodes)
    nodesf = nodes[is_orphan, :]
    pointsf = points[is_orphan, :]
    nodes = nodes[~is_orphan, :]
    points = points[~is_orphan, :]
    return nodes, points, nodesf, pointsf

def paraview_indexing(nodes, poly3, poly4):
    npt = nodes.shape[0]
    nodes_idx = np.arange(npt)
    nodes_with_idx = np.hstack([nodes, nodes_idx.reshape(-1, 1)])
    poly3c = poly3.flatten()
    j3 = [np.where(nodes_with_idx[:, 0] == val)[0][0] for val in poly3c]
    poly3i = np.array(j3).reshape(poly3.shape)
    poly4c = poly4.flatten()
    j4 = [np.where(nodes_with_idx[:, 0] == val)[0][0] for val in poly4c]
    poly4i = np.array(j4).reshape(poly4.shape)
    npl3 = poly3i.shape[0]
    npl4 = poly4i.shape[0]
    offset3 = 3 * np.arange(1, npl3 + 1)
    offset4 = poly3i.size + 4 * np.arange(1, npl4 + 1)
    offset = np.concatenate([offset3, offset4])
    return poly3i, poly4i, npl3, npl4, offset

def read_openfast_outb(outb_file):
    # Read OpenFAST binary output
    fastout = FASTOutputFile(outb_file)
    data = fastout.toDataFrame()
    # print(data.columns)
    # Extract 6 DOF columns
    dof_names = ['B1Surge_[m]', 'B1Sway_[m]', 'B1Heave_[m]', 'B1Roll_[rad]', 'B1Pitch_[rad]', 'B1Yaw_[rad]']
    missing = [name for name in dof_names if name not in data.columns]
    if missing:
        raise ValueError(f'Missing DOF columns in OpenFAST outb file: {missing}')
    tran = data[dof_names[:3]].values  # Surge, Sway, Heave
    rota = data[dof_names[3:]].values  # Roll, Pitch, Yaw
    time = data['Time_[s]'].values
    return tran, rota, time, data

def read_fst_for_vtkfps(fst_file):
    dt = None
    fps = None
    with open(fst_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2:
                if parts[1] == 'DT':
                    dt = float(parts[0])
                elif parts[1] == 'VTK_fps':
                    fps = float(parts[0])
                    break
    if fps is None or dt is None:
        raise ValueError("VISUALIZATION inputs including VTK_fps in '.fst' file are missing or wrong")
    # dt, fps 출력
    print(f"Read from FST: DT = {dt}, VTK_fps = {fps}")
    return dt, fps

def calculate_export_steps(time, dt, fps):
    dt_out = time[1] - time[0] if len(time) > 1 else dt
    dt_export_rate = round(1 / fps / dt)
    dt_export = dt_export_rate * dt
    t_export = np.arange(0, time[-1] + dt_export, dt_export)
    export_indices = [np.argmin(np.abs(time - t)) for t in t_export]
    return export_indices

def write_vtp(filename, points, poly3i, poly4i, offset, npl3, npl4, color=None):
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfVerts="0" NumberOfLines="0"\n')
        f.write(f'           NumberOfStrips="0" NumberOfPolys="{npl3 + npl4}">\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for pt in points:
            f.write(f'{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Polys>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for tri in poly3i:
            f.write(f'{tri[0]} {tri[1]} {tri[2]}\n')
        for quad in poly4i:
            f.write(f'{quad[0]} {quad[1]} {quad[2]} {quad[3]}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for off in offset:
            f.write(f'{off}\n')
        f.write('        </DataArray>\n')
        f.write('      </Polys>\n')
        # Optional: CellData for color
        if color is not None:
            f.write('      <CellData Scalars="Color">\n')
            f.write('        <DataArray type="UInt8" Name="Color" NumberOfComponents="3" format="ascii">\n')
            for c in color:
                f.write(f'{c[0]} {c[1]} {c[2]}\n')
            f.write('        </DataArray>\n')
            f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')

# def generate_pvsm(path_out, n_data):
#     # This is a minimal template for ParaView state file
#     pvsm_content = f'''<ParaView>
#   <ServerManagerState version="5.10.1">
#     <Proxy group="animation" type="AnimationScene" id="263" servers="16">
#       <Property name="EndTime" id="263.EndTime" number_of_elements="1">
#         <Element index="0" value="{n_data-1}"/>
#       </Property>
#       <Property name="NumberOfFrames" id="263.NumberOfFrames" number_of_elements="1">
#         <Element index="0" value="{n_data}"/>
#       </Property>
#       <!-- Add more animation and coloring properties as needed -->
#     </Proxy>
#     <!-- Add more proxies for sources, representations, coloring, etc. -->
#   </ServerManagerState>
# </ParaView>
# '''
#     pvsm_path = os.path.join(path_out, 'PlatformSurface.pvsm')
#     with open(pvsm_path, 'w') as f:
#         f.write(pvsm_content)
#     print(f'ParaView state file generated: {pvsm_path}')

def load_inputs(filename):
    input_cfg, output_cfg = read_config(filename)
    nastran_model_file = input_cfg['nastran_model_file']
    bdf_offset_str = input_cfg.get('bdf_offset', '0,0,0')
    bdf_offset = np.array([float(x) for x in bdf_offset_str.split(',')])
    print(f'bdf_offset: {bdf_offset}')

    OF_FST = input_cfg['OF_FST']
    OF_HD = input_cfg['OF_HD']
    openfast_outb = input_cfg['openfast_outb']
    return output_cfg, nastran_model_file, bdf_offset, OF_FST, OF_HD, openfast_outb

def rotation_matrix_zyx(roll, pitch, yaw):
    """
    ZYX 오일러 각(roll, pitch, yaw)에 대한 회전 행렬 반환
    roll: X축 회전 [rad]
    pitch: Y축 회전 [rad]
    yaw: Z축 회전 [rad]
    """
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    return np.array([
    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
    [-sp,   cp*sr,            cp*cr]
    ])

def main():
    # 1. 시작 시간 로깅
    dt_start = datetime.now()
    print(f"Start time: {dt_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. 설정 파일 읽기
    output_cfg, nastran_model_file, bdf_offset, \
        OF_FST, OF_HD, openfast_outb = load_inputs('input_data.ini')

    # 3. 출력 폴더 준비
    path_out = setup_output_folder(output_cfg['path_out'])

    # 4. Nastran bulk 데이터 파싱
    nodes, points, poly3, poly4 = parse_nastran_bulk(nastran_model_file)

    # 5. 무게중심 및 회전중심 좌표 추출
    PtfmRef = extract_ptfm_ref_coords(OF_HD) 

    # 7. Paraview 인덱싱
    poly3i, poly4i, npl3, npl4, offset = paraview_indexing(nodes, poly3, poly4)

    # 8. 좌표 변환
    points = points + bdf_offset

    # 9. OpenFAST 출력 읽기
    tran, rota, time, ofd = read_openfast_outb(openfast_outb)

    # 10. FST 파일에서 시간 및 프레임레이트 읽기
    dt, fps = read_fst_for_vtkfps(OF_FST)

    # 11. 내보낼 스텝 계산
    export_indices = calculate_export_steps(time, dt, fps)
    tran_export = tran[export_indices, :]
    rota_export = rota[export_indices, :]
    n_data = len(tran_export)
    n_data_order = 1 + int(np.floor(np.log10(n_data)))
    fname_format = f'PlatformSurface.%0{n_data_order}d.vtp'

    # 12. 내보내기 루프
    print('Exporting VTP sequence...')
    n_data = len(tran_export)
    for it in range(n_data):
        # Rigid body 변환: COG(PtfmRef) 기준 회전 후 이동
        roll, pitch, yaw = rota_export[it, :]

        # 회전변환 행렬 (ZYX 오일러 각: yaw, pitch, roll)
        rota_m = rotation_matrix_zyx(roll, pitch, yaw)

        points_centered = points - PtfmRef
        points_rot = (rota_m @ points_centered.T).T
        points_t = points_rot + PtfmRef
        points_t = points_t + tran_export[it, :]

        color = None
        fname = os.path.join(path_out, fname_format % it)
        write_vtp(fname, points_t, poly3i, poly4i, offset, npl3, npl4, color)

    # 13. ParaView 상태 파일 생성
    #generate_pvsm(path_out, n_data)

if __name__ == '__main__':
    main()
