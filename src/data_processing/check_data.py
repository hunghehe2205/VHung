import os

# ========== CẤU HÌNH ==========
DATA_DIR = "/home/emogenai4e/emo/Hung_data/UCF_Crime"
TRAIN_TXT = "/home/emogenai4e/emo/Hung_data/Anomaly_Train.txt"
TEST_TXT  = "/home/emogenai4e/emo/Hung_data/Anomaly_Test.txt"

# ========== ĐỌC FILE LIST ==========
def read_list(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

train_list = read_list(TRAIN_TXT)
test_list  = read_list(TEST_TXT)
all_files  = train_list + test_list

print(f"Train: {len(train_list)} | Test: {len(test_list)} | Tổng: {len(all_files)}")
print("="*50)

# ========== KIỂM TRA ==========
missing   = []
corrupted = []

for rel_path in all_files:
    full_path = os.path.join(DATA_DIR, rel_path.strip())
    
    if not os.path.exists(full_path):
        missing.append(rel_path)
    else:
        size = os.path.getsize(full_path)
        if size < 10 * 1024:  # < 10KB → nghi lỗi
            corrupted.append((rel_path, size))

# ========== KẾT QUẢ ==========
print(f"\n❌ THIẾU ({len(missing)} files):")
for f in missing:
    print(f"  {f}")

print(f"\n⚠️  NGHI LỖI ({len(corrupted)} files, size < 10KB):")
for f, s in corrupted:
    print(f"  {f}  ({s/1024:.1f} KB)")

print(f"\n✅ Có đủ: {len(all_files) - len(missing) - len(corrupted)} files")