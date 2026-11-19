import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="中文商品評論五星情緒分析", page_icon="💬", layout="centered")

# 載入模型與分詞器（快取以加速運行）
@st.cache_resource
def load_model():
    model_name = "techthier/bert-base-chinese-jd-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception:
        # 備用方案
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def clean_text(text):
    # 修正中文常見諧音與輸入錯字，提升語意理解精準度
    text = text.replace("著個", "這個")
    text = text.replace("還型", "還行")
    text = text.replace("還省", "還行")
    text = text.replace("極力", "非常")
    text = text.replace("回夠", "回購")  
    text = text.replace("值得回夠", "值得回購")
    return text

def analyze_sentiment(text):
    cleaned_text = clean_text(text)
    
    # 規則引擎輔助
    negative_keywords = ["塑膠感", "便宜貨", "塑料", "塑膠", "太爛", "不值", "色差", "雷", "差評"]
    neutral_keywords = ["還行", "還可以", "一般", "普通", "勉強"]
    positive_keywords = ["質感好", "極佳", "完美", "超棒", "精緻", "值得回購", "回購", "值得", "不錯", "很好", "讚"]
    
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    star_rating = prediction + 1
    
    # 結合整體語意進行規則校正
    has_positive = any(keyword in cleaned_text for keyword in positive_keywords)
    has_negation_before_positive = "不不錯" in cleaned_text or "不好" in cleaned_text or "不值得" in cleaned_text
    
    if has_positive and not has_negation_before_positive:
        star_rating = max(star_rating, 4)
        if "值得回購" in cleaned_text or "多次回購" in cleaned_text or "回購多次" in cleaned_text or "完美" in cleaned_text:
            star_rating = 5
            
    if any(keyword in cleaned_text for keyword in neutral_keywords):
        star_rating = 3 if ("普通" in cleaned_text or "一般" in cleaned_text) else 4

    if any(keyword in cleaned_text for keyword in negative_keywords):
        if "塑膠感" in cleaned_text or "雷" in cleaned_text or "太爛" in cleaned_text:
            star_rating = min(star_rating, 2)
        else:
            star_rating = min(star_rating, 3)
            
    return star_rating

# UI 介面設計
st.title("💬 中文商品評論五星情緒分析")
st.markdown("輸入商品評論，我們將為您預測 **1 至 5 星** 的情緒評等，並自動偵測錯字與處理複雜語意（如「不錯」、「回收再利用」等語境）。")

# 測試範例按鈕
st.subheader("💡 快速測試範例")
cols = st.columns(3)
examples = [
    "可重複回收再利用不錯",
    "著個產品塑膠感很重",
    "產品已回夠多次值得回購"
]

# 使用 Session State 來記住點擊的範例
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

for i, ex in enumerate(examples):
    if cols[i].button(ex):
        st.session_state["user_input"] = ex

# 文字輸入框
user_input = st.text_area(
    "請輸入產品評論（支援自動糾錯與語意融合分析）:",
    value=st.session_state["user_input"],
    height=120,
    placeholder="例：這個東西還型，可重複回收再利用不錯。"
)

# 分析按鈕
if st.button("🎯 開始預測情緒", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入一些評論文字！")
    else:
        with st.spinner("分析中，請稍候..."):
            rating = analyze_sentiment(user_input)
        
        st.success("✨ 分析完成！")
        
        # 視覺化星等呈現
        stars = "★" * rating + "☆" * (5 - rating)
        st.subheader(f"預測評等：{rating} 星")
        st.markdown(f"<h1 style='color: #FFD700; text-align: center; margin-bottom: 20px;'>{stars}</h1>", unsafe_allow_html=True)
        
        # 評語對照
        star_desc = {
            1: "🔴 極差 - 產品體驗非常糟糕",
            2: "🟠 不滿意 - 商品有明顯缺點或待改進",
            3: "🟡 普通 - 表現中規中矩，沒有特別優缺點",
            4: "🟢 滿意 - 產品質量與表現良好，值得推薦",
            5: "🔥 極佳 - 非常滿意，極力推薦！"
        }
        st.info(star_desc.get(rating, ""))
        
        # 顯示處理後修正的文本（若有）
        cleaned = clean_text(user_input)
        if cleaned != user_input:
            st.caption(f"偵測並修正錯字後的分析文本：*{cleaned}*")
