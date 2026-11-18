"""Insert measured results into the phase report; never fabricate missing runs."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[2]

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def main() -> None:
    out=ROOT/'outputs'/(ROOT/'outputs/LATEST.txt').read_text().strip()
    audit=json.loads((out/'artifact_audit.json').read_text())
    if audit['status']!='PASS': raise ValueError('Artifacts not verified')
    summary=read_csv(out/'test_summary.csv');bench=read_csv(out/'benchmark.csv')
    config=json.loads((out/'config_frozen_before_test.json').read_text())
    rel='../outputs/'+out.name
    report=ROOT/'docs/04_bao_cao_phan_thanh_vien_2.md'
    content=report.read_text(encoding='utf-8')
    pre=content.split('## 4. Thực nghiệm và kết quả')[0]
    pre=pre.replace('Chờ P4 để điền số đo.','Số đo P4 đã bổ sung bên dưới; mức tăng tốc chỉ áp dụng cấu hình và máy đã đo.')
    # Rebuilding is idempotent: keep the manually authored methods, replace generated tables.
    if '### 2.1. Kết quả đo thực tế' in pre:
        start=pre.index('### 2.1. Kết quả đo thực tế')
        end=pre.index('## 3. Kiểm thử chương trình',start)
        pre=pre[:start]+pre[end:]
    table='### 2.1. Kết quả đo thực tế\n\n'
    table+=f'Nguồn: [benchmark.csv]({rel}/benchmark.csv), [môi trường]({rel}/environment.json). Cùng ảnh validation Poisson đầu tiên, 256×256; 7 lần đo sau warmup, lấy median. K/lambda đúng cấu hình đã chọn.\n\n'
    table+='| Thuật toán | Baseline ms | Optimized ms | Tăng tốc | Traced MiB trước → sau | Sai khác cực đại |\n|---|---:|---:|---:|---:|---:|\n'
    for r in bench:
        table+=f"| {r['method']} | {float(r['baseline_s'])*1000:.3f} | {float(r['optimized_s'])*1000:.3f} | {float(r['speedup']):.2f}× | {float(r['baseline_traced_mib']):.2f} → {float(r['optimized_traced_mib']):.2f} | {float(r['max_abs_error']):.2e} |\n"
    table+=f'\n![Benchmark]({rel}/benchmark.png)\n\n'
    table+='Phần thay đổi nằm trong `solve`: nhánh `optimized`, cặp FFT thuận/nghịch và vị trí tính H/H liên hợp trước vòng lặp. Các vòng lặp theo pixel được thay bằng phép toán mảng ngay từ baseline; vì vậy bảng không tách riêng lợi ích vector hóa. Hai thay đổi real FFT và cache được đo chung, chưa xác định đóng góp riêng từng thay đổi. Tracemalloc đo cấp phát được theo dõi, không phải peak RSS. Chưa có benchmark Java/Fiji.\n\n'
    pre=pre.replace('## 3. Kiểm thử chương trình',table+'## 3. Kiểm thử chương trình')
    testlog=sorted((ROOT/'outputs/verification').glob('unittest_*.log'))[-1]
    pre+=f'\nLog thực chạy: [unittest]({testlog.relative_to(ROOT).as_posix().replace("outputs/","../outputs/",1)}). Audit artifact: [{audit["metric_rows"]} bản ghi metric]({rel}/artifact_audit.json), tái tính từ TIFF float32 khớp CSV trong tolerance 1e-4.\n\n'
    body='## 4. Thực nghiệm và kết quả\n\n'
    body+=f'Lần chạy: `{out.name}`. [Manifest]({rel}/manifest.json), [config khóa trước test]({rel}/config_frozen_before_test.json), [sweep validation]({rel}/validation_sweep.csv), [CSV test từng ảnh]({rel}/test_metrics.csv).\n\n'
    body+='### 4.1. Chọn tham số\n\n5 ảnh validation dùng chọn cấu hình theo PSNR trung bình sau clip [0,1]. Không dùng 10 ảnh test để chỉnh tham số. NIF giữ floor=1e-4; TRIF lambda ∈ {0.0001,0.001,0.01,0.02}; RIF lambda ∈ {0.0001,0.001,0.005,0.02}; RL K ∈ {5,10,20,40}; RLTV K=20, delta=0.01, lambda ∈ {0.0001,0.001}.\n\n'
    body+='| Nhiễu | Thuật toán | Cấu hình chọn từ validation |\n|---|---|---|\n'
    for noise,methods in config['selected'].items():
        for method,params in methods.items(): body+=f'| {noise} | {method} | `{json.dumps(params)}` |\n'
    body+=f'\n![Validation sweep]({rel}/validation_sweep.png)\n\n'
    body+='### 4.2. Kết quả test cố định\n\nBảng dưới là trung bình trên 10 ảnh, PSNR kèm độ lệch chuẩn mẫu, SSIM trung bình. Cùng data_range=1 và clip [0,1] cho các phương pháp. CSV chứa thêm điểm raw để thấy tác động của clipping; ảnh TIFF luôn giữ raw.\n\n'
    body+='| Nhiễu | Phương pháp | PSNR trung bình ± SD (dB) | SSIM |\n|---|---|---:|---:|\n'
    for r in summary: body+=f"| {r['noise']} | {r['method']} | {float(r['psnr_mean']):.3f} ± {float(r['psnr_std']):.3f} | {float(r['ssim_mean']):.4f} |\n"
    body+='\n'
    for noise in ['gaussian','poisson']:
        rs=[r for r in summary if r['noise']==noise]
        observed=next(r for r in rs if r['method']=='observed')
        best=max((r for r in rs if r['method']!='observed'),key=lambda r:float(r['psnr_mean']))
        body+=f"Với {noise}, {best['method']} có PSNR trung bình cao nhất trong các cấu hình này: {float(best['psnr_mean']):.3f} dB, chênh {float(best['psnr_mean'])-float(observed['psnr_mean']):+.3f} dB so với ảnh quan sát. Đây là kết quả của subset và lưới tham số đã chọn.\n\n"
    body+='NIF khuếch đại các thành phần có đáp ứng PSF nhỏ; floor hạn chế chia gần 0 nhưng không thay thế điều chuẩn. TRIF/RIF giảm khuếch đại nhiễu bằng phạt độ lớn/độ cong. RL dùng dừng sớm; RLTV thêm ưu tiên biến thiên nhỏ, có thể làm mất texture. PSNR và SSIM đo hai khía cạnh khác nhau, cần đọc cả bảng và ảnh. Không diễn giải phương pháp thắng trong bảng là luôn tốt hơn trên mọi ảnh.\n\n'
    body+=f'![Quality]({rel}/quality_summary.png)\n\n'
    body+=f'![Ví dụ test Gaussian]({rel}/test/100007_gaussian/comparison.png)\n\n![ROI]({rel}/test/100007_gaussian/roi.png)\n\n'
    body+=f'[Gallery đầy đủ 20 trường hợp, có đầu vào và năm đầu ra]({rel}/gallery.html). ROI luôn là vùng tâm 64×64; không chọn theo kết quả đẹp.\n\n'
    body+='### 4.3. Giới hạn\n\nĐây là mô phỏng 2D với PSF biết trước, không phải khử mờ mù hay ảnh kính hiển vi thật. JPEG tham chiếu đã có nén/gamma; Gaussian cắt âm làm thay đổi mô hình Gaussian thuần. Cùng toán tử tuần hoàn được dùng để tạo và khôi phục nên bài thử thuận lợi hơn khi PSF thực sai lệch. Chỉ 5 validation + 10 test theo tên, không đại diện toàn bộ BSDS500; số đo không phải benchmark segmentation chính thức. Chưa thử nhiều seed, độ mờ, PSF sai lệch hoặc chứng minh hội tụ RLTV.\n\n'
    body+='## 5. Tái lập và bàn giao\n\n'
    body+='Đọc [README mã nguồn](../code/README.md) để chạy lại. Các bước: `prepare_data.py` → `check.py` → `run_experiments.py` → `audit_artifacts.py`. Mỗi run có thư mục riêng; `outputs/LATEST.txt` chỉ run mới nhất đã xuất xong. Có thể mở gallery.html trực tiếp để xem ảnh; không cần Jupyter hay dependency mới.\n\n'
    body+='Đối chiếu Fiji: **chưa thực hiện**. Bàn giao `reference.tif`, `observed.tif`, `psf.tif`, `input.json`, `config_frozen_before_test.json`; TV3 lưu output plugin, tham số, log và ảnh giao diện. Không gọi số liệu Python là số liệu Fiji. Mã nhóm và phần ghép báo cáo/slides toàn nhóm còn do nhóm hoàn thiện.\n\n'
    body+='## Tài liệu tham khảo\n\n- Nguyễn Minh Hiếu, báo cáo lý thuyết nội bộ, công thức (8), (13), (15)–(24).\n- [Arbelaez et al., Contour Detection and Hierarchical Image Segmentation, TPAMI 2011; BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).\n- [Sage et al., DeconvolutionLab2, Methods 2017](https://bigwww.epfl.ch/deconvolution/deconvolutionlab2/sage1701p.pdf), mục 3.9 cho RLTV.\n\n'
    body+='## Nhật ký phase\n\n- P0: kế hoạch BSDS500 và khung ba mục được tạo trước triển khai.\n- P1: tải archive Berkeley; kiểm tra 1125 thành viên; 15 ảnh và SHA256. Tải chậm nên P2/P3 thực hiện trong lúc tải; chưa chạy test dataset trước manifest.\n- P2: năm solver, metrics, baseline và optimized; cập nhật công thức và quy ước.\n- P3: 18 test PASS sau sửa delta TV; đã lưu log.\n- P4: chọn tham số trên validation, khóa config, xuất test/benchmark; audit TIFF/CSV PASS.\n- P5: chờ rà soát cuối tài liệu và gallery.\n'
    report.write_text(pre+body,encoding='utf-8')
    print('Updated report from verified artifacts')

if __name__=='__main__': main()
