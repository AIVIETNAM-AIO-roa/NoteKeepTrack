import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

# Custom css cho đẹp
current_dir = os.path.dirname(os.path.abspath(__file__)) # Thư mục 'pages'
project_root = os.path.dirname(current_dir)             # Thư mục gốc 'NoteKeepTrack'
css_path = os.path.join(project_root, "css", "style.css") # Đường dẫn chuẩn đến file CSS
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error(f"Không tìm thấy file CSS tại đường dẫn: {css_path}")

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Keeptrack day 6",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: TRÌNH MÔ PHỎNG THAM SỐ ---
st.sidebar.header("LLM Parameters Playground")
st.sidebar.markdown("Thử nghiệm thay đổi các tham số để xem phân bố xác suất thay đổi theo thời gian thực.")


with st.sidebar.container(border=True):
    temperature = st.slider(
        "🌡️ Temperature (T)", 
        min_value=0.01, max_value=2.0, value=1.0, step=0.05,
        help="Thấp (gần 0): Tập trung, chính xác. Cao (gần 2): Sáng tạo, ngẫu nhiên."
    )
    top_p = st.slider(
        "🎯 Top P (Nucleus Sampling)",
        min_value=0.1, max_value=1.0, value=1.0, step=0.05,
        help="Chỉ giữ lại nhóm từ có tổng xác suất cộng dồn đạt ngưỡng P."
    )

simulate_btn = st.sidebar.button("🎲 Mô phỏng sinh chữ (Generate)")
# --- NỘI DUNG CHÍNH (MAIN CONTENT) ---
st.markdown(
    '<h1 class="gradient-title">Foundation of Prompt Engineering</h1>',
    unsafe_allow_html=True
)
st.markdown("---")

# Khởi tạo tabs cho 2 mục lớn
tab1, tab2 = st.tabs(["📊 Tổng quan về Prompt Engineering", "📝 Prompt Technique"])

with tab1:
    st.header("Các tham số cài đặt mô hình (LLM Settings)")
    st.markdown("""
    Việc tinh chỉnh tham số là bước không thể thiếu để kiểm soát đầu ra của LLM sao cho phù hợp với từng loại bài toán.
    """)
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.subheader("1. Temperature (Nhiệt độ)")
        st.markdown("""
        Đây là tham số quan trọng nhất, điều khiển độ "mượt mà" của hàm softmax, từ đó quyết định tính ngẫu nhiên hay ổn định của mô hình.
        """)
        
        st.latex(r"P(i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}")
        
        st.markdown("""
        *   **Trường hợp T = 1.0:** Không có sự thay đổi, giống như áp dụng hàm softmax bình thường.
        *   **Trường hợp T > 1.0:** Đồ thị được "làm mượt" (phẳng hơn). Các từ có xác suất thấp vốn ít được chọn nay có cơ hội xuất hiện cao hơn. Nếu set quá cao (>1.5), kết quả có thể ngẫu nhiên hoàn toàn và vô nghĩa.
        *   **Trường hợp T < 1.0:** Đồ thị được "làm sắc nét". Mô hình sẽ tập trung chọn các từ có xác suất cao.
        *   **Trường hợp T ứng với 0.0:** Phân bố xác suất bị đẩy mức tối đa (xác suất = 1 cho từ cao nhất). Mô hình luôn chọn từ có xác suất cao nhất (Greedy Decoding), mang lại kết quả cố định và đáng tin cậy nhất.
        
        > 💡 Dùng `T=0.0` cho bài toán cần độ chính xác (toán, logic, code); `T=0.7 - 0.8` cho bài toán cần viết lách sáng tạo.
        """)
        
        st.subheader("2. Top P (Nucleus Sampling)")
        st.markdown("""
        Giúp điều chỉnh độ đa dạng bằng cách chỉ chọn ra nhóm các từ có tổng xác suất cộng dồn vượt qua một ngưỡng P.
        
        > 💡 Dùng P thấp (0.1 - 0.5) cho câu trả lời chính xác, P cao (0.6 - 0.9) cho câu trả lời sáng tạo. **Lưu ý:** Không nên thay đổi cùng lúc cả Temperature và Top p.
        """)
        
        st.subheader("3. Kiểm soát độ dài và lặp từ")
        st.markdown("""
        *   **Max Length:** Giới hạn số lượng token tối đa được sinh ra, giúp kiểm soát chi phí.
        *   **Stop Sequences:** Thiết lập chuỗi ký tự dừng (Ví dụ đặt `11.` để dừng khi danh sách đạt 10 mục).
        *   **Frequency Penalty (0.5 - 1.0):** Phạt từ ngữ dựa trên *số lần xuất hiện*, giúp giảm lặp cụm từ.
        *   **Presence Penalty (0.6 - 1.0):** Phạt đồng đều miễn là từ đó *đã xuất hiện*, thúc đẩy mô hình mở rộng ý tưởng mới.
        """)

    with col2:
        st.subheader("📊 Biểu đồ phân bố xác suất trực quan")
        st.info(f"Cấu hình hiện tại: **Temperature = {temperature}** | **Top P = {top_p}**")
        
        # --- XỬ LÝ TOÁN HỌC MÔ PHỎNG ---
        tokens = ["AI", "Học_tập", "Mô_hình", "Dữ_liệu", "Công_nghệ"]
        logits = np.array([4.0, 3.2, 2.5, 1.8, 0.8])
        
        # Áp dụng Temperature vào Logits
        scaled_logits = logits / temperature
        # Tính Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # Trừ max để tránh tràn số
        probs = exp_logits / np.sum(exp_logits)
        
        # Áp dụng cơ chế lọc Top P (Nucleus Sampling)
        # Sắp xếp giảm dần để tính tổng tích lũy
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)
        
        # Tìm các token được giữ lại (Tổng tích lũy trước đó < Top P)
        allowed_indices = []
        for idx, cum_p in enumerate(cum_probs):
            allowed_indices.append(sorted_indices[idx])
            if cum_p >= top_p:
                break
                
        # Tạo mảng màu sắc: xanh cho token hợp lệ, xám mờ cho token bị loại bởi Top P
        colors = ['#1f77b4' if i in allowed_indices else '#d3d3d3' for i in range(len(tokens))]
        
        # --- VẼ BIỂU ĐỒ BẰNG PLOTLY ---
        fig = go.Figure(data=[go.Bar(
            x=tokens,
            y=probs,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='auto',
        )])
        
        fig.update_layout(
            yaxis=dict(title="Xác suất chọn (Probability)", range=[0, 1.05]),
            xaxis=dict(title="Các Tokens (Từ ngữ mẫu)"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Các cột màu xanh đại diện cho các từ nằm trong vùng chọn của Top P. Cột xám bị loại bỏ.")
        
        # --- MÔ PHỎNG OUTPUT ---
        if simulate_btn:
            st.subheader("🎲 Kết quả mô phỏng sinh chữ:")
            # Chuẩn hóa lại xác suất của các từ được giữ lại sau khi lọc Top P
            filtered_probs = np.zeros_like(probs)
            filtered_probs[allowed_indices] = probs[allowed_indices]
            filtered_probs /= np.sum(filtered_probs)
            
            chosen_token = np.random.choice(tokens, p=filtered_probs)
            
            st.success(f"Mô hình đã chọn từ: **'{chosen_token}'**")
            if temperature > 1.2:
                st.warning("Mức temperature cao! Mô hình đang có xu hướng chọn các từ ngẫu nhiên và 'bay bổng' hơn.")
            elif temperature < 0.4:
                st.info("Mức temperature thấp! Mô hình hoạt động rất kỷ luật và ưu tiên từ có xác suất cao nhất.")

with tab2:    
    st.header("Bảng Tổng Hợp Kiến Thức Kỹ Thuật Prompt Engineering")
    st.markdown("""
    Bảng phân loại chi tiết 15 kỹ thuật tương tác với LLM dựa trên bộ câu hỏi đánh giá cốt lõi.
    """)
    
    # Định nghĩa cấu trúc bảng dữ liệu
    data = [
        {
            "Kỹ thuật": "Zero-Shot Prompting",
            "Nhóm": "Instruction",
            "Vấn đề giải quyết": "Thực hiện nhiệm vụ chưa được huấn luyện trực tiếp bằng cách mô tả bằng ngôn ngữ tự nhiên.",
            "Template tối thiểu": "[Instruction] + [Input data] -> [Output].",
            "Yếu tố phụ thuộc": "Phụ thuộc vào cấu trúc rõ ràng của câu lệnh.",
            "Trường hợp không nên dùng": "Bài toán suy luận phức tạp, tính toán nhiều bước."
        },
        {
            "Kỹ thuật": "Few-Shot Prompting",
            "Nhóm": "Instruction",
            "Vấn đề giải quyết": "Giúp mô hình học cách thực hiện nhiệm vụ dựa trên một số ít ví dụ cụ thể trong quá trình suy luận.",
            "Template tối thiểu": "[Ví dụ 1] + [Ví dụ 2] + [Input thực tế] -> [Output].",
            "Yếu tố phụ thuộc": "Chất lượng, thứ tự, và phân phối nhãn của các ví dụ mẫu.",
            "Trường hợp không nên dùng": "Bài toán đòi hỏi tư duy logic vượt qua khả năng học vẹt qua mẫu."
        },
        {
            "Kỹ thuật": "Chain-of-Thought (CoT)",
            "Nhóm": "Reasoning",
            "Vấn đề giải quyết": "Cải thiện hiệu suất giải quyết bài toán phức tạp bằng cách tạo chuỗi suy nghĩ trung gian trước khi ra kết quả.",
            "Template tối thiểu": "[Câu hỏi] + \"Let's think step by step\".",
            "Yếu tố phụ thuộc": "Temperature = 0.0 và chất lượng lập luận trong ví dụ mẫu.",
            "Trường hợp không nên dùng": "Câu hỏi tra cứu thông tin đơn giản."
        },
        {
            "Kỹ thuật": "Auto-CoT",
            "Nhóm": "Automation / Reasoning",
            "Vấn đề giải quyết": "Tự động hóa việc tạo ra các ví dụ mẫu chứa chuỗi suy nghĩ, giảm sự can thiệp thủ công.",
            "Template tối thiểu": "Phân cụm -> Zero-shot CoT sinh rationale -> Tổng hợp thành Few-shot CoT.",
            "Yếu tố phụ thuộc": "Khả năng tự suy luận của LLM để tự sinh chuỗi lập luận ban đầu.",
            "Trường hợp không nên dùng": "Khi mô hình gốc suy luận yếu, dễ lan truyền lỗi."
        },
        {
            "Kỹ thuật": "Self-Consistency",
            "Nhóm": "Reasoning / Optimization",
            "Vấn đề giải quyết": "Khắc phục lỗi suy luận lẻ tẻ của CoT bằng cách sinh đa dạng hướng suy nghĩ và chọn đáp án phổ biến nhất.",
            "Template tối thiểu": "[Dùng CoT sinh N đường suy luận] -> [Lấy majority vote].",
            "Yếu tố phụ thuộc": "Temperature (> 0 để đa dạng) và chi phí token.",
            "Trường hợp không nên dùng": "Ngân sách token bị hạn hẹp hoặc bài toán đơn giản."
        },
        {
            "Kỹ thuật": "Generated Knowledge",
            "Nhóm": "Reasoning",
            "Vấn đề giải quyết": "Khắc phục việc thiếu kiến thức nền bằng cách sinh ra các phát biểu kiến thức liên quan trước khi trả lời.",
            "Template tối thiểu": "Sinh kiến thức -> Tích hợp kiến thức vào câu hỏi -> Trả lời.",
            "Yếu tố phụ thuộc": "Độ tin cậy và chính xác của kiến thức được mô hình sinh ra ở bước 1.",
            "Trường hợp không nên dùng": "Khi lĩnh vực dễ khiến LLM bị ảo giác (hallucination) thông tin."
        },
        {
            "Kỹ thuật": "Tree of Thoughts (ToT)",
            "Nhóm": "Reasoning",
            "Vấn đề giải quyết": "Hỗ trợ ra quyết định phức tạp qua việc phân rã trạng thái, đánh giá và tìm kiếm (BFS/DFS).",
            "Template tối thiểu": "Phân rã -> Sinh ứng viên -> Đánh giá Heuristic -> Tìm kiếm BFS/DFS.",
            "Yếu tố phụ thuộc": "Hàm đánh giá Heuristic và thuật toán tìm kiếm.",
            "Trường hợp không nên dùng": "Tác vụ một bước đơn giản, không cần quay lui (backtracking)."
        },
        {
            "Kỹ thuật": "Automatic Prompt Engineer (APE)",
            "Nhóm": "Optimization / Automation",
            "Vấn đề giải quyết": "Tự động hóa tìm ra Prompt (Instruction) tối ưu thay vì đoán thủ công.",
            "Template tối thiểu": "LLM sinh Instruction -> Đánh giá điểm -> Lọc top K prompt tốt nhất -> Resample.",
            "Yếu tố phụ thuộc": "Tập dữ liệu mẫu (train subset) và hàm đánh giá (score function).",
            "Trường hợp không nên dùng": "Tác vụ dùng một lần, thiếu tập dữ liệu mẫu để chấm điểm."
        },
        {
            "Kỹ thuật": "Active-Prompt",
            "Nhóm": "Optimization",
            "Vấn đề giải quyết": "Tối ưu hóa chọn ví dụ bằng cách tìm câu hỏi có độ bất định cao nhất để con người gán nhãn ưu tiên.",
            "Template tối thiểu": "LLM dự đoán -> Tính độ bất định -> Chọn câu khó -> Gán nhãn thủ công.",
            "Yếu tố phụ thuộc": "Công thức đo độ bất định (Entropy, Variance) và bước gán nhãn thủ công.",
            "Trường hợp không nên dùng": "Khi không có nhân sự chuyên môn để gán nhãn thủ công."
        },
        {
            "Kỹ thuật": "Directional Stimulus Prompting",
            "Nhóm": "Instruction / Optimization",
            "Vấn đề giải quyết": "Điều khiển LLM tạo nội dung đi đúng hướng mong muốn bằng các kích thích chỉ dẫn (từ khóa).",
            "Template tối thiểu": "[Văn bản gốc] + [Hướng dẫn: Hint/Keywords] -> [Output].",
            "Yếu tố phụ thuộc": "Mô hình nhỏ được fine-tune để tự động sinh ra các từ khóa định hướng.",
            "Trường hợp không nên dùng": "Tác vụ không yêu cầu tóm tắt hay sinh nội dung bám sát từ khóa."
        },
        {
            "Kỹ thuật": "PAL (Program-aided)",
            "Nhóm": "Agent/Tool",
            "Vấn đề giải quyết": "Khắc phục yếu điểm tính toán sai của LLM bằng cách chuyển bước suy luận thành code (Python).",
            "Template tối thiểu": "[Câu hỏi] -> [Sinh code lập luận] -> [Chạy qua Interpreter] -> [Kết quả].",
            "Yếu tố phụ thuộc": "Môi trường thông dịch mã bên ngoài và khả năng code chuẩn cú pháp.",
            "Trường hợp không nên dùng": "Bài toán văn chương, sáng tạo không có thuật toán, số liệu."
        },
        {
            "Kỹ thuật": "ReAct",
            "Nhóm": "Agent/Tool",
            "Vấn đề giải quyết": "Tránh ảo giác bằng cách kết hợp suy luận (Thought) với hành động tra cứu môi trường ngoài (Action).",
            "Template tối thiểu": "Lặp lại: [Thought] -> [Action] -> [Observation] -> [Thought].",
            "Yếu tố phụ thuộc": "Stop Sequences và sự ổn định của công cụ tra cứu (ví dụ: Wikipedia API).",
            "Trường hợp không nên dùng": "Mô hình đã có sẵn toàn bộ kiến thức để trả lời mà không cần tra cứu thêm."
        },
        {
            "Kỹ thuật": "Reflexion",
            "Nhóm": "Agent/Tool / Optimization",
            "Vấn đề giải quyết": "Giúp đại diện AI tự học qua văn bản tự phản tư, đánh giá lỗi và lưu vào bộ nhớ.",
            "Template tối thiểu": "Actor -> Evaluator -> Self-Reflection -> Lưu Memory -> Lặp lại.",
            "Yếu tố phụ thuộc": "Hàm đánh giá chất lượng và hệ thống bộ nhớ ngắn hạn (Memory).",
            "Trường hợp không nên dùng": "Các tác vụ ngắn, hỏi đáp một lần vì quy trình này tốn nhiều token và thời gian."
        },
        {
            "Kỹ thuật": "Multimodal CoT",
            "Nhóm": "Multimodal",
            "Vấn đề giải quyết": "Tạo chuỗi suy luận tận dụng tối đa dữ liệu đa phương tiện khi đầu vào có cả hình ảnh và chữ.",
            "Template tối thiểu": "(1) Ảnh + Chữ -> Sinh Rationale -> (2) Ảnh + Chữ + Rationale -> Đáp án.",
            "Yếu tố phụ thuộc": "Kiến trúc mạng neural có khả năng trích xuất đặc trưng hình ảnh.",
            "Trường hợp không nên dùng": "Các bài toán chỉ xử lý bằng văn bản thuần túy (text-only)."
        },
        {
            "Kỹ thuật": "Synthetic Prompting",
            "Nhóm": "Optimization / Reasoning",
            "Vấn đề giải quyết": "Dùng LLM tự tổng hợp thêm ví dụ mẫu đa dạng qua quy trình backward-forward để giảm công gán nhãn.",
            "Template tối thiểu": "Ví dụ gốc -> Backward (tạo câu hỏi) -> Forward (tạo chuỗi lập luận) -> Lọc.",
            "Yếu tố phụ thuộc": "Chất lượng ví dụ mồi ban đầu và độ phong phú của chủ đề gốc.",
            "Trường hợp không nên dùng": "Đã có sẵn bộ dữ liệu huấn luyện lớn được gán nhãn đầy đủ."
        }
    ]
    df = pd.DataFrame(data)
    
    # Hiển thị dữ liệu prompt dưới dạng bảng đẹp mắt
    st.table(df)

    # Tạo một hộp ghi chú mẹo hay
    st.info("💡 Có thể giữ một file 'Prompt Template' cá nhân lưu trữ các khung xương vai trò cấu trúc sẵn để tái sử dụng nhanh chóng cho các tác vụ hàng ngày!")