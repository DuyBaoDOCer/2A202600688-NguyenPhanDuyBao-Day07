# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Phan Duy Bảo
**Mã số sinh viên:** 2A202600688
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai text chunk có high cosine similarity nghĩa là vector embedding của chúng trỏ theo hướng gần nhau trong không gian vector — tức là chúng có nội dung ngữ nghĩa tương đồng, bất kể độ dài hay scale của văn bản.

**Ví dụ HIGH similarity:**
- Sentence A: "The cat sat on the mat."
- Sentence B: "A cat is resting on the mat."
- Tại sao tương đồng: Cả hai câu đều mô tả con mèo và tấm thảm, dùng động từ liên quan đến tư thế đứng yên → nghĩa ngữ nghĩa gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: "I love eating pizza."
- Sentence B: "The stock market crashed today."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (ẩm thực vs. tài chính), không có từ ngữ hay khái niệm chung nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector, bỏ qua độ lớn, nên nó không bị ảnh hưởng bởi độ dài văn bản — một tài liệu dài hơn sẽ có embedding vector lớn hơn nhưng cosine similarity vẫn phản ánh đúng mức độ tương đồng về nghĩa. Euclidean distance lại bị chi phối bởi độ lớn vector nên sẽ cho kết quả sai lệch khi so sánh văn bản có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks**`
>
> Overlap lớn hơn làm bước nhảy (step) nhỏ lại, tạo ra nhiều chunk hơn. Người ta muốn overlap nhiều hơn để đảm bảo context quan trọng nằm ở ranh giới giữa hai chunk không bị cắt đứt — giúp retrieval không bỏ sót thông tin nằm "ở giữa" hai chunk liên tiếp.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Điều khoản dịch vụ

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain 'Điều khoản dịch vụ' vì đặc thù văn bản có tính cấu trúc chặt chẽ, ngôn từ pháp lý phức tạp và dễ bị mất ngữ cảnh nếu chunking sai. Tập dữ liệu này rất lý tưởng để đánh giá độ chính xác của các thuật toán phân tách cấu trúc, đồng thời cung cấp các thuộc tính rõ ràng để thử nghiệm khả năng truy xuất kết hợp Metadata Filtering

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | be_tos.txt | be.com.vn | 3,541 | source=be, type=tos, language=vi |
| 2 | grab_tos.txt | grab.com/vn | 223,216 | source=grab, type=tos, language=vi |
| 3 | shopee_tos.txt | shopee.vn | 82,685 | source=shopee, type=tos, language=vi |
| 4 | viettelpost_tos.txt | viettelpost.com.vn | 14,361 | source=viettelpost, type=tos, language=vi |
| 5 | zalopay_tos.txt | zalopay.vn | 21,726 | source=zalopay, type=tos, language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "grab", "shopee", "viettelpost" | Filter theo nền tảng khi query hỏi về một công ty cụ thể — tránh retrieve chunk từ ToS của Shopee khi hỏi về ViettelPost |
| type | string | "tos" | Phân biệt với file dạng khác nếu mở rộng thêm loại tài liệu (e.g. FAQ, policy) |
| language | string | "vi" | Hữu ích khi tập dữ liệu đa ngôn ngữ; có thể filter language=vi để tránh lẫn chunk tiếng Anh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên đoạn văn bản mẫu AI/ML (~265 ký tự × 3 = ~795 ký tự), `chunk_size=100`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| AI/ML sample | FixedSizeChunker (`fixed_size`) | 8 | 99.38 | Thấp — cắt giữa câu |
| AI/ML sample | SentenceChunker (`by_sentences`) | 5 | 158.2 | Cao — giữ trọn vẹn câu |
| AI/ML sample | RecursiveChunker (`recursive`) | 15 | 51.27 | Trung bình — nhỏ hơn chunk_size |

### Strategy Của Tôi

**Loại:** `SentenceChunker` (max_sentences_per_chunk=3)

**Mô tả cách hoạt động:**
> Strategy dùng regex `(?<=[.!?])\s+|(?<=\.)\n` để detect ranh giới câu (sau dấu `.`, `!`, `?` có khoảng trắng hoặc xuống dòng). Sau đó gom nhóm tối đa 3 câu liên tiếp thành một chunk, join lại bằng khoảng trắng và strip whitespace thừa. Mỗi chunk đảm bảo kết thúc ở ranh giới câu tự nhiên nên vẫn đọc được hoàn chỉnh.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> SentenceChunker phù hợp với văn bản dạng FAQ, policy hoặc tài liệu prose vì mỗi câu thường mang một ý trọn vẹn. Chunk giữ nguyên cấu trúc câu nên embedding sẽ capture đúng nghĩa hơn so với fixed-size cắt giữa từ. Với top_k=3, retrieval sẽ trả về 3 "đoạn ý nghĩa" thay vì 3 mảnh character ngẫu nhiên.

**Code snippet (implementation):**
```python
def chunk(self, text: str) -> list[str]:
    if not text:
        return []
    sentence_pattern = re.compile(r'(?<=[.!?])\s+|(?<=\.)\n')
    raw_sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    for i in range(0, len(sentences), self.max_sentences_per_chunk):
        group = sentences[i : i + self.max_sentences_per_chunk]
        chunk_text = " ".join(group).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| AI/ML sample | fixed_size (baseline) | 8 | 99.38 | Thấp — cắt giữa câu |
| AI/ML sample | **SentenceChunker (của tôi)** | 5 | 158.2 | Cao — câu hoàn chỉnh |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Phan Duy Bảo (Tôi) | SentenceChunker (max=3) | 0/10 (MockEmbedder) | Câu hoàn chỉnh, giữ nguyên ngữ cảnh ngữ pháp lý. | Kích thước chunk không đồng đều, MockEmbedder không hỗ trợ semantic. |
| Trần Nguyễn Anh Thư | Sliding Window with overlap (size=300, overlap=50) | 0/10 (MockEmbedder) | | Dễ cắt ngang câu/từ ở ranh giới chunk, làm mất context pháp lý. |
| Lê Hữu Khoa | RecursiveChunker (size=400) | 0/10 (MockEmbedder) | Khai thác ranh giới tự nhiên (ưu tiên tách theo đoạn \n\n), giúp chunk trùng khớp với một điều khoản hoàn chỉnh. | Kích thước chunk biến động lớn giữa các phân đoạn. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Đối với domain "Điều khoản dịch vụ" có tính cấu trúc pháp lý chặt chẽ theo các điều, khoản và câu văn rất dài phức tạp, **RecursiveChunker** hoặc **SentenceChunker** là những lựa chọn tốt nhất.
> - **RecursiveChunker** giúp ngắt văn bản ở cấp độ đoạn (`\n\n` hoặc `\n`), bảo đảm giữ nguyên vẹn nội dung của cả một mục hoặc một điều khoản mà không bị đứt đoạn.
> - **SentenceChunker** giữ nguyên vẹn ranh giới câu pháp lý, tránh bị đứt mạch thông tin nghĩa vụ/quyền lợi.
> - **Sliding Window with overlap (FixedSizeChunker)** là lựa chọn kém hiệu quả nhất vì việc cắt cứng theo số lượng ký tự thường xuyên cắt đôi các câu pháp lý quan trọng ngay giữa chừng, làm đứt đoạn ngữ nghĩa mặc dù có cơ chế overlap để bù đắp một phần context.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex lookbehind `(?<=[.!?])\s+` để split sau dấu câu kết thúc mà không mất dấu đó (khác với `re.split(r'[.!?]\s')` làm mất dấu câu). Sau đó dùng sliding window cố định là `max_sentences_per_chunk` câu, join bằng space. Edge case: text rỗng trả về `[]`, text không có câu nào (không có dấu câu) trả về toàn bộ text trong một chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm greedy merge: với mỗi separator theo thứ tự ưu tiên, split text -> gom liên tiếp các piece nhỏ vào buffer cho đến khi thêm piece tiếp theo sẽ vượt `chunk_size` -> flush buffer thành chunk, recurse trên piece quá lớn với separator kế tiếp. Base case: text ≤ chunk_size -> trả về ngay; hết separator -> character-level slice.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi `Document` được chuyển thành record gồm `{id, content, embedding, metadata}` trong `_make_record`. `add_documents` append vào `self._store` (in-memory list). `search` embed query rồi tính dot product với mọi embedding đã lưu, sort descending, trả về top-k. Dùng dot product thay vì cosine vì embedding từ `MockEmbedder` đã được normalize (unit vectors) -> dot product = cosine similarity.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` **pre-filter trước**: lọc `self._store` bằng list comprehension kiểm tra từng key-value trong `metadata_filter`, sau đó gọi `_search_records` trên subset đã lọc. `delete_document` dùng list comprehension tạo list mới loại bỏ tất cả record có `metadata["doc_id"] == doc_id`, trả về `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt structure theo RAG chuẩn: System instruction -> Context (numbered chunks, kèm source) -> Question -> "Answer:". Context inject rõ ràng số thứ tự `[1]`, `[2]`, `[3]` và tên nguồn để LLM có thể trích dẫn. Instruction "Only use the context below" giúp giảm hallucination. Fallback message khi store rỗng: "No relevant documents found."

### Test Results

```
============================================ test session starts =============================================
platform win32 -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\duybaoDOCer\miniconda3\envs\vinuni\python.exe
cachedir: .pytest_cache
rootdir: D:\vinuni\Day_7\2A202600688-NguyenPhanDuyBao-Day07
plugins: anyio-4.13.0, langsmith-0.8.8
collected 42 items                                                                                            

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                   [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                            [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                     [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                      [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                           [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED           [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                 [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                  [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                  [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                  [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                             [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                         [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                   [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED          [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED              [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED        [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED              [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                  [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                    [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                      [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                            [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                 [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                   [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED       [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                    [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                             [ 64%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED              [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                  [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                        [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                  [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED             [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED            [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED           [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED    [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================================= 42 passed in 0.11s =============================================
```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Sử dụng `all-MiniLM-L6-v2` (dim=384, real semantic embedder).

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "The cat sat on the mat." | "A cat is resting on the mat." | high | **0.8821** | ✅ Rất cao — đúng kỳ vọng |
| 2 | "Python is a programming language." | "Python is used for machine learning." | high | **0.7911** | ✅ Cao — đúng kỳ vọng |
| 3 | "I love eating pizza." | "The stock market crashed today." | low | **0.0179** | ✅ Gần 0 — đúng kỳ vọng |
| 4 | "Neural networks are inspired by the brain." | "Deep learning uses layered neural networks." | high | **0.5168** | ✅ Trung bình cao |
| 5 | "The sun rises in the east." | "Photosynthesis requires sunlight." | medium | **0.3437** | ✅ Thấp-trung bình |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Với `all-MiniLM-L6-v2`, tất cả 5 dự đoán đều đúng chiều — tương phản hoàn toàn với MockEmbedder (0/5 đúng). Pair 1 đạt 0.8821, xác nhận model hiểu "cat resting" ≈ "cat sat". Pair 3 (pizza vs. stock market) chỉ đạt 0.0179 — gần như không liên quan. Pair 4 bất ngờ nhất: "Neural networks" vs "Deep learning" chỉ đạt 0.5168 (không cao như kỳ vọng), vì model nhận ra đây là hai khái niệm *liên quan* nhưng không đồng nhất. Bài học quan trọng: real embedder phản ánh ngữ nghĩa thực sự — MockEmbedder hoàn toàn không có giá trị cho semantic retrieval.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Grab yêu cầu người dùng phải đủ bao nhiêu tuổi để ký kết hợp đồng? | 18 tuổi. Không thể ký kết hợp đồng nếu dưới 18 tuổi (Điều 3.1.1) |
| 2 | Thời hạn giải quyết khiếu nại của ViettelPost là bao lâu? | Không quá 2 tháng với dịch vụ trong nước; 3 tháng với dịch vụ quốc tế (Điều 9b) |
| 3 | Thời hiệu khiếu nại về bưu gửi bị mất của ViettelPost là bao lâu? | 6 tháng kể từ ngày kết thúc thời gian toàn trình (Điều 9a) |
| 4 | Shopee có thể xóa tài khoản người dùng vì những lý do nào? | Tài khoản không hoạt động, vi phạm điều khoản, hành vi lừa đảo/quấy rối, nhiều tài khoản, lạm dụng mã giảm giá, thông tin giả mạo (Điều 5.4) |
| 5 | Nếu không đồng ý với thay đổi điều khoản của Grab, người dùng cần làm gì? | Hủy tài khoản hoặc xóa ứng dụng. Tiếp tục sử dụng = đồng ý (Điều 1.3) |

### Kết Quả Của Tôi

Embedder: `all-MiniLM-L6-v2` (local, dim=384). Scores là cosine similarity thực tế.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Grab yêu cầu người dùng phải đủ bao nhiêu tuổi để ký kết hợp đồng? | Grab — Gói TMĐT, điều khoản ký hợp đồng (`grab_chunk_387`) | **0.7335** | Partial | Có nhắc đến hợp đồng nhưng không chỉ rõ 18 tuổi |
| 2 | Thời hạn giải quyết khiếu nại ViettelPost? | ViettelPost — không quá 03 tháng với bưu chính quốc tế (`viettelpost_chunk_33`) | **0.6790** | **Yes** ✅ | Đúng: ≤2 tháng nội địa, ≤3 tháng quốc tế |
| 3 | Thời hiệu khiếu nại bưu gửi bị mất? | Grab — GrabExpress, đoạn về bưu gửi (`grab_chunk_275`) | **0.6033** | Partial | Nhắc đến bưu gửi nhưng của GrabExpress, không phải ViettelPost |
| 4 | Shopee xóa tài khoản vì lý do nào? | Shopee — đoạn về Tài Khoản Phụ và quyền của Shopee (`shopee_chunk_31`) | **0.7473** | Partial | Nhắc đến xóa tài khoản nhưng thiếu danh sách lý do đầy đủ |
| 5 | Không đồng ý thay đổi điều khoản Grab? | Grab — GrabXu, điều khoản sửa đổi (`grab_chunk_134`) | **0.6735** | No | Sai nguồn — nói về GrabXu thay vì Điều 1.3 chính |

**Bao nhiêu queries trả về chunk relevant trong top-3?**
- Q2: ✅ `viettelpost_chunk_33` (rank 1) + `viettelpost_chunk_32` (rank 2) — **2/3 relevant**
- Q4: ✅ `shopee_chunk_31` (rank 1) — **liên quan đến xóa tài khoản**
- Q1, Q3, Q5: top-3 không có chunk chứa gold answer chính xác

**Tổng: 2 / 5 queries có top-1 relevant** (so với 0/5 với MockEmbedder)

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách thiết kế và tận dụng Metadata Filtering hiệu quả từ các bạn. Bằng việc phân chia tài liệu ToS theo các trường metadata có tính lọc cao như `source` và `category`, hệ thống có thể pre-filter trước khi chạy similarity search, loại bỏ hoàn toàn nhiễu từ các nền tảng khác nhau và nâng cao độ chính xác của kết quả.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác đã demo việc tích hợp mô hình embedding thực tế (như `sentence-transformers` nội địa hoặc OpenAI `text-embedding-3-small`) thay vì chỉ dùng mock embeddings. Việc này làm thay đổi hoàn toàn chất lượng tìm kiếm, chuyển từ các kết quả ngẫu nhiên và không liên quan sang tìm kiếm ngữ nghĩa chính xác hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thử `RecursiveChunker` với cấu trúc separators ưu tiên `["\n\n", "\n", ". "]` để chia nhỏ tài liệu ToS theo đúng ranh giới của các Điều, Khoản pháp lý (thường phân tách bằng dấu xuống dòng hoặc ký tự mục lục). Bên cạnh đó, tôi chắc chắn sẽ sử dụng một mô hình embedding thực tế có hỗ trợ tiếng Việt để đánh giá đúng năng lực của RAG agent.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
