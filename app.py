import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import json
import io
import re
import requests
from dotenv import load_dotenv
import pandas as pd
from streamlit_paste_button import paste_image_button
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="메뉴판 자동 분류 시스템", layout="wide")
load_dotenv()

# 사이드바 설정
with st.sidebar:
    st.title("설정")
    
    # API 키: Streamlit Secrets > .env > 수동 입력
    env_api_key = None
    try:
        env_api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass
    
    if not env_api_key:
        env_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not env_api_key:
        api_key_input = st.text_input("Google API Key를 입력하세요", type="password")
        if api_key_input:
            env_api_key = api_key_input
            st.success("API 키가 입력되었습니다.")
        else:
            st.error("API 키가 설정되지 않았습니다.")
    else:
        st.success("API 키가 로드되었습니다.")
    
    # 모델 선택
    selected_model = st.selectbox(
        "사용할 모델 선택",
        ["gemini-2.5-flash-preview-05-20", "gemini-1.5-flash", "직접 입력"],
        index=0
    )
    
    if selected_model == "직접 입력":
        model_name = st.text_input("커스텀 모델 이름", placeholder="예: gemini-1.5-pro")
    else:
        model_name = selected_model

if env_api_key:
    genai.configure(api_key=env_api_key)

# ==========================================
# 2. 네이버 메뉴 이미지 가져오기 (requests 기반, Playwright 불필요)
# ==========================================
def fetch_naver_menu_images(url):
    """네이버 플레이스 메뉴 페이지에서 이미지 URL을 추출 (requests + 정규식)"""
    # URL에서 place ID 추출
    place_id_match = re.search(r'place/(\d+)', url) or re.search(r'restaurant/(\d+)', url)
    
    if not place_id_match:
        st.error("URL에서 플레이스 ID를 찾을 수 없습니다. URL을 확인해주세요.")
        return []
    
    place_id = place_id_match.group(1)
    
    # 네이버 플레이스 API 엔드포인트 (공개 데이터)
    api_url = f"https://pcmap-api.place.naver.com/graphql"
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://pcmap.place.naver.com/restaurant/{place_id}/menu/list",
        "Origin": "https://pcmap.place.naver.com"
    }
    
    # GraphQL 쿼리로 메뉴 데이터 요청
    payload = [
        {
            "operationName": "getRestaurant",
            "variables": {"restaurantId": place_id, "input": {"deviceType": "pc"}},
            "query": "query getRestaurant($restaurantId: String!, $input: RestaurantInput) { restaurant(id: $restaurantId, input: $input) { id name menus { name price description images } } }"
        }
    ]
    
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=15)
        data = resp.json()
        
        # GraphQL 응답에서 메뉴 이미지 추출 시도
        menus = data[0].get("data", {}).get("restaurant", {}).get("menus", [])
        if menus:
            image_urls = []
            for menu in menus:
                for img in menu.get("images", []):
                    if isinstance(img, str):
                        image_urls.append(img)
            if image_urls:
                return image_urls
    except Exception:
        pass
    
    # 폴백: pcmap 페이지를 requests로 가져와서 이미지 URL 패턴 추출
    try:
        target_url = f"https://pcmap.place.naver.com/restaurant/{place_id}/menu/list"
        resp = requests.get(target_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }, timeout=15)
        
        # HTML/JS 소스에서 이미지 URL 패턴 매칭
        img_pattern = re.findall(r'https?://[^\s"\']*pstatic\.net[^\s"\']*(?:type=f[^\s"\']*|menu[^\s"\']*)', resp.text)
        if img_pattern:
            return list(dict.fromkeys(img_pattern))[:10]
        
        # 더 넓은 범위의 pstatic 이미지
        img_pattern2 = re.findall(r'https?://[^\s"\'<>]*pstatic\.net/[^\s"\'<>]+\.(?:jpg|png|jpeg)', resp.text)
        if img_pattern2:
            return list(dict.fromkeys(img_pattern2))[:10]
            
    except Exception:
        pass
    
    return []

# ==========================================
# 3. AI 로직
# ==========================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception)
)
def call_gemini_vision(prompt, image, model_id):
    model = genai.GenerativeModel(model_id)
    # 이미지 포맷 보정
    if image is not None and getattr(image, 'format', None) is None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image = Image.open(buf)
    response = model.generate_content([prompt, image])
    return response.text

def extract_menu_items_from_images(images, model_id):
    """이미지에서 메뉴 리스트를 추출"""
    prompt = """이 메뉴판 이미지에서 메뉴 이름과 가격을 추출해서 JSON 리스트 형식으로만 답변해줘.
    형식: [{"이름": "메뉴명", "가격": "가격"}]
    반드시 JSON 데이터만 출력해."""
    try:
        model = genai.GenerativeModel(model_id)
        # 이미지 포맷 보정
        processed = []
        for img in images:
            if img is None or not hasattr(img, 'save'):
                continue
            if getattr(img, 'format', None) is None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                processed.append(Image.open(buf))
            else:
                processed.append(img)
        
        if not processed:
            st.error("분석할 수 있는 유효한 이미지가 없습니다.")
            return pd.DataFrame()
        
        response = model.generate_content([prompt] + processed)
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "[" in text:
            text = text[text.find("["):text.rfind("]")+1]
        return pd.DataFrame(json.loads(text))
    except Exception as e:
        st.error(f"메뉴 분석 실패: {e}")
        return pd.DataFrame()

def classify_food(food_image, menu_list, model_id):
    menu_str = ", ".join(menu_list)
    prompt = f"다음 메뉴 리스트 중 이 사진과 가장 잘 어울리는 메뉴 하나만 골라줘: [{menu_str}]. 없으면 'Others'."
    try:
        if food_image is None:
            return "Others"
        result = call_gemini_vision(prompt, food_image, model_id).strip()
        for menu in menu_list:
            if menu in result: return menu
        return "Others"
    except: return "Others"

# ==========================================
# 4. UI 구성
# ==========================================
st.title("메뉴판 자동 분류 시스템")

# 세션 관리
if 'menu_df' not in st.session_state: st.session_state.menu_df = pd.DataFrame()
if 'menu_images' not in st.session_state: st.session_state.menu_images = []

# 1단계: 스토어명
store_name = st.text_input("1. 스토어명을 입력하세요", placeholder="예: 맛나식당_본점")
st.markdown("---")

# 2단계: 메뉴판 로드
st.subheader("2. 메뉴판 불러오기")
tab1, tab2 = st.tabs(["네이버 메뉴 주소", "이미지 붙여넣기"])

with tab1:
    naver_url = st.text_input("네이버 플레이스 URL", placeholder="https://map.naver.com/...")
    if st.button("메뉴 불러오기") and naver_url:
        with st.spinner("네이버에서 메뉴 이미지를 가져오는 중..."):
            urls = fetch_naver_menu_images(naver_url)
            if urls:
                new_images = []
                for u in urls[:5]:
                    try:
                        resp = requests.get(u, stream=True, timeout=10)
                        new_images.append(Image.open(resp.raw))
                    except: pass
                if new_images:
                    st.session_state.menu_images = new_images
                    st.success(f"{len(new_images)}개 이미지 로드 완료")
                else:
                    st.warning("이미지 다운로드에 실패했습니다.")
            else:
                st.warning("이미지를 찾지 못했습니다. URL을 확인해주세요.")

with tab2:
    pasted = paste_image_button("이미지 붙여넣기 (클릭 후 Ctrl+V)", key="paste_btn")
    if pasted and pasted.image_data:
        st.session_state.menu_images = [pasted.image_data]
        st.image(pasted.image_data, caption="붙여넣은 이미지", width=300)

# 3단계: AI 분석
if st.session_state.menu_images:
    st.image(st.session_state.menu_images, width=150)
    if st.button("메뉴 데이터 추출 시작", type="primary"):
        with st.spinner("AI가 메뉴를 분석 중..."):
            st.session_state.menu_df = extract_menu_items_from_images(st.session_state.menu_images, model_name)
            if not st.session_state.menu_df.empty:
                st.success("메뉴 추출 완료!")

# 4단계: 결과 확인 및 분류
if not st.session_state.menu_df.empty:
    st.subheader("3. 추출된 메뉴 정보 확인")
    st.session_state.menu_df = st.data_editor(st.session_state.menu_df, num_rows="dynamic", use_container_width=True)
    
    st.markdown("---")
    st.subheader("4. 분류할 사진 업로드")
    food_files = st.file_uploader("분류할 음식 사진들을 선택하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("자동 분류 실행"):
        if not store_name:
            st.error("스토어명을 입력해주세요!")
        elif not food_files:
            st.error("음식 사진을 선택해주세요!")
        else:
            progress = st.progress(0)
            base_path = f"./{store_name}"
            os.makedirs(base_path, exist_ok=True)
            
            menu_list = st.session_state.menu_df["이름"].tolist()
            results = []
            
            for i, f in enumerate(food_files):
                img = Image.open(f)
                cat = classify_food(img, menu_list, model_name)
                save_dir = os.path.join(base_path, cat)
                os.makedirs(save_dir, exist_ok=True)
                img.save(os.path.join(save_dir, f.name))
                results.append((f.name, cat))
                progress.progress((i+1)/len(food_files))
            
            st.success(f"'{store_name}' 폴더에 분류가 완료되었습니다!")
            with st.expander("결과 상세 내역"):
                st.table(pd.DataFrame(results, columns=["파일명", "분류 메뉴"]))

with st.expander("도움말"):
    st.write("""
    - **1단계**: 스토어명을 입력합니다.
    - **2단계**: 네이버 URL 또는 이미지 붙여넣기로 메뉴 정보를 가져옵니다.
    - **3단계**: AI가 메뉴를 추출하면 테이블에서 확인/수정합니다.
    - **4단계**: 음식 사진을 업로드하고 실행하면 자동 분류됩니다.
    """)