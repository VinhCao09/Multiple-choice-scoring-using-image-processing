import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict
import matplotlib.pyplot as plt

# function lấy tọa độ x từ khối
def lay_toa_do_x(khoi):
    return khoi[1][0]

# function lấy tọa độ y từ khối
def lay_toa_do_y(khoi):
    return khoi[1][1]

# function get chiều cao từ khối
def lay_chieu_cao(khoi):
    return khoi[1][3]

# tính diện tích để sắp xếp contour
def tinh_dien_tich(khoi):
    x, y, w, h = cv2.boundingRect(khoi)
    return x * y

# cắt ảnh để tìm các khối đáp án
def cat_anh_tim_khoi(anh):
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    anh_lam_mo = cv2.GaussianBlur(anh_xam, (5, 5), 0)
    anh_canh = cv2.Canny(anh_lam_mo, 100, 200)
    duong_vien = cv2.findContours(anh_canh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    duong_vien = imutils.grab_contours(duong_vien)

    danh_sach_khoi = []
    x_truoc, y_truoc, w_truoc, h_truoc = 0, 0, 0, 0

    if duong_vien:
        duong_vien = sorted(duong_vien, key=tinh_dien_tich)
        for dv in duong_vien:
            x, y, w, h = cv2.boundingRect(dv)
            if w * h > 100000:
                khoang_cach_min = x * y - x_truoc * y_truoc
                khoang_cach_max = (x + w) * (y + h) - (x_truoc + w_truoc) * (y_truoc + h_truoc)
                if not danh_sach_khoi or (khoang_cach_min > 20000 and khoang_cach_max > 20000):
                    danh_sach_khoi.append((anh_xam[y:y+h, x:x+w], (x, y, w, h)))
                    x_truoc, y_truoc, w_truoc, h_truoc = x, y, w, h
        return sorted(danh_sach_khoi, key=lay_toa_do_x)
    return []

# trích xuất các dòng đáp án từ các khối
def trich_xuat_dong(danh_sach_khoi):
    cac_dong = []
    vi_tri_dong = []

    for khoi_anh, (x_khoi, y_khoi, w_khoi, h_khoi) in danh_sach_khoi:
        mang_khoi = np.array(khoi_anh)
        buoc_doc = ceil(mang_khoi.shape[0] / 6)
        for i in range(6):
            anh_hop = np.array(mang_khoi[i * buoc_doc:(i + 1) * buoc_doc, :])
            chieu_cao_hop = anh_hop.shape[0]
            anh_hop = anh_hop[14:chieu_cao_hop - 14, :]
            buoc_ngang = ceil(anh_hop.shape[0] / 5)
            for j in range(5):
                cac_dong.append(anh_hop[j * buoc_ngang:(j + 1) * buoc_ngang, :])
                y_dong = y_khoi + i * buoc_doc + j * buoc_ngang + 14
                vi_tri_dong.append((x_khoi, y_dong, buoc_ngang))
    return cac_dong, vi_tri_dong

# ưxử lý các dòng để lấy các ô đáp án trên ảnhw
def xu_ly_cac_dong(cac_dong):
    danh_sach_o = []
    khoang_cach = 44
    diem_bat_dau = 32

    for anh_dong in cac_dong:
        for i in range(4):
            o_dap_an = anh_dong[:, diem_bat_dau + i * khoang_cach:diem_bat_dau + (i + 1) * khoang_cach]
            o_dap_an = cv2.threshold(o_dap_an, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            o_dap_an = cv2.resize(o_dap_an, (28, 28), cv2.INTER_AREA)
            o_dap_an = o_dap_an.reshape((28, 28, 1))
            danh_sach_o.append(o_dap_an)
    
    if len(danh_sach_o) != 480:
        raise ValueError("Số lượng ô đáp án phải là 480")
    return danh_sach_o

# ánh xạ chỉ số thành đáp án (A, B, C, D)
def anh_xa_dap_an(chi_so):
    if chi_so % 4 == 0:
        return "A"
    elif chi_so % 4 == 1:
        return "B"
    elif chi_so % 4 == 2:
        return "C"
    else:
        return "D"

#dự đoán đáp án của học sinh
def du_doan_dap_an(danh_sach_o):
    ket_qua = defaultdict(list)
    mo_hinh = CNN_Model('weight.h5').build_model(rt=True)
    mang_o = np.array(danh_sach_o)
    diem_so = mo_hinh.predict_on_batch(mang_o / 255.0)

    for idx, diem in enumerate(diem_so):
        cau_hoi = idx // 4
        if diem[1] > 0.9:
            dap_an_chon = anh_xa_dap_an(idx)
            ket_qua[cau_hoi + 1].append(dap_an_chon)
    return ket_qua

# func đọc đáp án đúng từ file .txt
def doc_dap_an_dung(tap_tin):
    dap_an_dung = {}
    with open(tap_tin, 'r', encoding='utf-8') as f:
        for dong in f:
            cau_hoi, dap_an = dong.strip().split(':')
            dap_an_dung[int(cau_hoi)] = dap_an.strip()
    return dap_an_dung

# func check đáp án của học sinh
def kiem_tra_dap_an(dap_an_hs, dap_an_dung):
    ket_qua_kiem_tra = {}
    for cau_hoi, dap_an_dung_cau in dap_an_dung.items():
        dap_an_chon = dap_an_hs.get(cau_hoi, [])
        if len(dap_an_chon) == 1 and dap_an_chon[0] == dap_an_dung_cau:
            ket_qua_kiem_tra[cau_hoi] = True
        else:
            ket_qua_kiem_tra[cau_hoi] = False
    return ket_qua_kiem_tra

# func tính điểm
def tinh_diem(ket_qua_kiem_tra):
    so_cau_dung = sum(1 for ket_qua in ket_qua_kiem_tra.values() if ket_qua)
    tong_so_cau = len(ket_qua_kiem_tra)
    diem = (so_cau_dung / tong_so_cau) * 10
    return so_cau_dung, diem

# function tô màu đáp án trên ảnh
def ve_to_mau_dap_an(anh, dap_an_hs, dap_an_dung, vi_tri_dong):
    anh_mau = anh.copy()
    khoang_cach = 44
    diem_bat_dau = 32
    chieu_rong_o = khoang_cach

    for cau_hoi in range(1, len(vi_tri_dong) + 1):
        x_khoi, y_dong, chieu_cao_dong = vi_tri_dong[cau_hoi - 1]
        y_giua = y_dong + chieu_cao_dong // 2

        dap_an_dung_cau = dap_an_dung.get(cau_hoi, '')
        dap_an_chon = dap_an_hs.get(cau_hoi, [])

        # dap an dung:
        if dap_an_dung_cau:
            chi_so_dung = ord(dap_an_dung_cau) - ord('A')
            x_dung = x_khoi + diem_bat_dau + chi_so_dung * khoang_cach + chieu_rong_o // 2
            cv2.circle(anh_mau, (x_dung, y_giua), 10, (255,90, 0), 2)  # Xanh dương

        # dap an hs chon:
        for dap_an in dap_an_chon:
            chi_so_chon = ord(dap_an) - ord('A')
            x_chon = x_khoi + diem_bat_dau + chi_so_chon * khoang_cach + chieu_rong_o // 2
            if dap_an == dap_an_dung_cau and len(dap_an_chon) == 1:
                cv2.circle(anh_mau, (x_chon, y_giua), 10, (0, 255, 0), 2)  # màu xanh
            else:
                cv2.circle(anh_mau, (x_chon, y_giua), 10, (0, 0, 255), 2)  # Màu đỏ

    return anh_mau

# function hien thi anh = matplotlib
def hien_thi_ket_qua(anh_mau, so_cau_dung, diem, tong_so_cau):
    anh_rgb = cv2.cvtColor(anh_mau, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 15))
    plt.imshow(anh_rgb)
    plt.title(f"Kq: {so_cau_dung}/{tong_so_cau} - Điểm: {diem:.2f}/10", fontsize=14)
    plt.axis('off')
    plt.show()


def main():
    anh = cv2.imread('1.jpg')
    if anh is None:
        raise FileNotFoundError("Không thể tải ảnh '1.jpg'. Vui lòng kiểm tra đường dẫn.")

    # tìm các khối đáp án
    danh_sach_khoi = cat_anh_tim_khoi(anh)
    print(f"Số khối đáp án: {len(danh_sach_khoi)}")

    # trích xuất các dòng đáp án
    cac_dong, vi_tri_dong = trich_xuat_dong(danh_sach_khoi)
    print(f"Số dòng đáp án: {len(cac_dong)}")

    danh_sach_o = xu_ly_cac_dong(cac_dong)
    dap_an_hs = du_doan_dap_an(danh_sach_o)
    print("Đáp án học sinh:", dict(dap_an_hs))

 
    dap_an_dung = doc_dap_an_dung('correct_answers.txt')
    ket_qua_kiem_tra = kiem_tra_dap_an(dap_an_hs, dap_an_dung)
    so_cau_dung, diem = tinh_diem(ket_qua_kiem_tra)
    print(f"Số câu đúng: {so_cau_dung}/{len(dap_an_dung)}")
    print(f"Điểm: {diem:.2f}/10")


    anh_mau = ve_to_mau_dap_an(anh, dap_an_hs, dap_an_dung, vi_tri_dong)

    hien_thi_ket_qua(anh_mau, so_cau_dung, diem, len(dap_an_dung))

if __name__ == '__main__':
    main()