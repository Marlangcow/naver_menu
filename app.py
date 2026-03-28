import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import shutil
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
import json
import base64
import io
import asyncio
from dotenv import load_dotenv
import streamlit.components.v1 as components
import pandas as pd
from playwright.async_api import async_playwright
from streamlit_paste_button import paste_image_button
import re
import subprocess
import sys

# ==========================================
# 0. 배포 환경 설정 (Playwright 브라우저 설치)
# ==========================================
def install_playwright():
    try:
        # 이미 설치되어 있는지 확인
        from playwright.sync_api import sync_playwright
    except ImportError:
        # 라이브러리 자체가 없으면 건너뜀 (requirements.txt에 의해 설치될 것)
        return

    # 브라우저 설치 시도 (Streamlit Cloud에서 첫 실행 시 필요)
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        print(f"Playwright install error: {e}")

# 실행
install_playwright()

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="음식 사진 자동 분류 시스템", layout="wide")
load_dotenv()

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    env_api_key = os.getenv("GOOGLE_API_KEY")
    if not env_api_key:
        st.error("❌ .env 파일에 GOOGLE_API_KEY가 설정되지 않았습니다.")
    else:
        st.success("✅ API 키가 로드되었습니다.")
    
    # 모델 선택 및 커스텀 입력
    selected_model = st.selectbox(
        "사용할 모델 선택",
        ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite-preview-09-2025", "직접 입력"],
        index=0
    )
    
    if selected_model == "직접 입력":
        model_name = st.text_input("커스텀 모델 이름을 입력하세요", placeholder="예: gemini-1.5-flash")
    else:
        model_name = selected_model

if env_api_key:
    genai.configure(api_key=env_api_key)

# ==========================================
# 2. 스크래퍼 (네이버 지도)
# ==========================================

async def scrape_naver_menu_images(url):
    """Playwright를 사용하여 네이버 플레이스 메뉴 이미지를 추출 (URL 정규화 포함)"""
    # 1. URL에서 플레이스 ID 추출 (정규표현식)
    place_id_match = re.search(r'place/(\d+)', url)
    if not place_id_match:
        # 다른 형식의 URL 시도 (예: pcmap 주소 직송)
        place_id_match = re.search(r'restaurant/(\d+)', url)
    
    if place_id_match:
        place_id = place_id_match.group(1)
        # 2. 다이렉트 메뉴 페이지로 정규화 (아이프레임 회피)
        target_url = f"https://pcmap.place.naver.com/restaurant/{place_id}/menu/list"
    else:
        # ID를 못 찾으면 입력받은 주소 그대로 사용 시도
        target_url = url

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 모바일 환경처럼 User-Agent 설정 (더 로드가 빠를 수 있음)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        )
        page = await context.new_page()
        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            
            # 3. 여러 가능한 셀렉터 시도
            selectors = [".place_section_content", ".menu_list", "._3y_Yt", "._19S90"]
            found_selector = None
            for s in selectors:
                try:
                    await page.wait_for_selector(s, timeout=5000)
                    found_selector = s
                    break
                except: continue
            
            if not found_selector:
                # 만약 지정된 셀렉터가 없으면 전체 페이지 로드 대기
                await page.wait_for_timeout(3000)
                found_selector = "body"

            # 4. 스크롤 로딩 유도 (Lazy Loading 이미지 캡처용)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            await page.wait_for_timeout(1000)
            
            # 5. 이미지 태그 추출
            images = await page.query_selector_all(f"{found_selector} img")
            srcs = []
            for img in images:
                src = await img.get_attribute("src")
                # 네이버 메뉴 이미지는 보통 pstatic.net 또는 네이버 클라우드 저장소 주소
                if src and ("pstatic.net" in src or "naver.com" in src):
                    srcs.append(src)
            
            # 6. 중복 제거
            srcs = list(dict.fromkeys(srcs))
            return srcs
        except Exception as e:
            st.error(f"스크래핑 중 오류 발생: {e}")
            return []
        finally:
            await browser.close()

# ==========================================
# 3. AI 로직 (Gemini API 활용)
# ==========================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_gemini_vision(prompt, image, model_id):
    model = genai.GenerativeModel(model_id)
    response = model.generate_content([prompt, image])
    return response.text

def extract_menu_items_from_images(images, model_id):
    """여러 이미지(메뉴판)에서 메뉴 리스트를 테이블 형식에 맞게 추출"""
    prompt = """
    이 메뉴판 이미지(들)에서 메뉴 이름과 가격을 추출해서 JSON 리스트 형식으로만 답변해줘.
    예: [{"이름": "김치찌개", "가격": "9,000원", "설명": "매콤한 맛"}, ...]
    오직 JSON 리스트만 출력하고 다른 설명은 하지 마.
    """
    try:
        model = genai.GenerativeModel(model_id)
        # PIL 이미지의 format이 None인 경우 Gemini API에서 오류가 발생할 수 있음
        processed_images = []
        for img in images:
            if img is None:
                continue
            # PIL Image 객체인지 확인
            if not hasattr(img, 'save'):
                continue
                
            if getattr(img, 'format', None) is None:
                # 임시 버퍼를 통해 포맷을 강제로 설정 (PNG)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                processed_images.append(Image.open(buf))
            else:
                processed_images.append(img)
                
        if not processed_images:
            st.error("분석할 수 있는 유효한 이미지가 없습니다.")
            return pd.DataFrame()
            
        response = model.generate_content([prompt] + processed_images)
        result_text = response.text
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "[" in result_text:
            result_text = result_text[result_text.find("["):result_text.rfind("]")+1]
        
        data = json.loads(result_text)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"메뉴 추출 중 오류 발생: {e}")
        return pd.DataFrame()

def classify_food(food_image, menu_list, model_id):
    menu_str = ", ".join(menu_list)
    prompt = f"다음 메뉴 중 가장 적합한 것을 고르세요: [{menu_str}]. 없으면 'Others'. 답변은 오직 메뉴 이름 하나만."
    try:
        # 이미지 포맷 보정 및 유효성 검사
        if food_image is None:
            return "Others"
            
        if getattr(food_image, 'format', None) is None:
            buf = io.BytesIO()
            food_image.save(buf, format="PNG")
            buf.seek(0)
            food_image = Image.open(buf)
            
        result = call_gemini_vision(prompt, food_image, model_id).strip()
        if result in menu_list: return result
        for menu in menu_list: 
            if menu in result or result in menu: return menu
        return "Others"
    except Exception: return "Others"

# ==========================================
# 4. UI/UX 구성 (5단계 프로세스)
# ==========================================

st.title("🍱 메뉴판 자동 분류 시스템")
st.markdown("---")

# 세션 상태 초기화
if 'menu_df' not in st.session_state: st.session_state.menu_df = pd.DataFrame()
if 'menu_images' not in st.session_state: st.session_state.menu_images = []

# 1단계: 스토어명
# store_name = st.text_input("1. 스토어명을 입력하세요", placeholder="예: 전설의우대갈비 마곡점")

# 2단계: 메뉴판 정보 연동
st.subheader("메뉴판 불러오기")
tab1, tab2 = st.tabs(["네이버 메뉴 주소", "메뉴판 이미지 붙여넣기"])

with tab1:
    naver_url = st.text_input("네이버 플레이스 메뉴 URL을 입력하세요", placeholder="https://map.naver.com/.../menu/list")
    if st.button("🌐 메뉴 불러오기") and naver_url:
        with st.spinner("네이버 지도에서 메뉴 이미지를 가져오는 중..."):
            image_urls = asyncio.run(scrape_naver_menu_images(naver_url))
            if image_urls:
                import requests
                new_images = []
                for url in image_urls:
                    try:
                        resp = requests.get(url, stream=True)
                        new_images.append(Image.open(resp.raw))
                    except: pass
                st.session_state.menu_images = new_images
                st.success(f"{len(st.session_state.menu_images)}개의 이미지를 불러왔습니다.")
            else:
                st.warning("이미지를 찾지 못했습니다. URL을 확인하거나 클래스명을 확인해주세요.")

with tab2:
    pasted_image = paste_image_button(
        label="📋 여기에 이미지 붙여넣기 (클릭 후 Ctrl+V)",
        key="paste_field"
    )
    if pasted_image and pasted_image.image_data:
        st.session_state.menu_images = [pasted_image.image_data]
        st.image(st.session_state.menu_images[0], caption="붙여넣은 이미지", width=300)

# 3단계: 메뉴 추출 시작
if st.session_state.menu_images and st.button("🔍 메뉴 데이터 추출 시작", type="primary"):
    with st.spinner("AI가 메뉴판 이미지를 분석하여 데이터로 변환 중..."):
        st.session_state.menu_df = extract_menu_items_from_images(st.session_state.menu_images, model_name)
        if not st.session_state.menu_df.empty:
            st.success("메뉴 추출 완료!")

# 4단계: 메뉴 정보 테이블 표시
if not st.session_state.menu_df.empty:
    st.subheader("4. 추출된 메뉴 정보 (확인 및 수정)")
    # 사용자가 테이블을 직접 수정할 수 있게 함
    st.session_state.menu_df = st.data_editor(st.session_state.menu_df, num_rows="dynamic", use_container_width=True)
    
    # 5단계: 사진 분류를 위한 업로드 섹션 표시
    st.markdown("---")
    st.subheader("5. 분류할 음식 사진들 업로드 및 실행")
    food_files = st.file_uploader("분류할 실제 음식 사진들을 선택하세요.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("🚀 자동 분류 및 저장 실행"):
        if not store_name:
            st.error("스토어명을 먼저 입력해주세요.")
        elif not food_files:
            st.error("음식 사진을 선택해주세요.")
        else:
            progress_bar = st.progress(0)
            base_path = f"./{store_name}"
            os.makedirs(base_path, exist_ok=True)
            
            menu_list = st.session_state.menu_df["이름"].tolist()
            results = []
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                def process_task(f):
                    img = Image.open(f)
                    cat = classify_food(img, menu_list, model_name)
                    save_dir = os.path.join(base_path, cat)
                    os.makedirs(save_dir, exist_ok=True)
                    img.save(os.path.join(save_dir, f.name))
                    return f.name, cat
                
                for i, (fname, cat) in enumerate(executor.map(process_task, food_files)):
                    results.append((fname, cat))
                    progress_bar.progress((i + 1) / len(food_files))
            
            st.success(f"✅ 분류 완료! '{store_name}' 폴더에 저장되었습니다.")
            with st.expander("결과 상세 내역"):
                st.table(pd.DataFrame(results, columns=["파일명", "분류 메뉴"]))

# 도움말
with st.expander("❓ 도움말 및 안내"):
    st.write("""
    - **1단계**: 작업할 폴더의 이름이 될 스토어명을 입력합니다.
    - **2단계**: 이미지 파일, 복사된 이미지, 또는 네이버 지도 URL 중 하나로 메뉴 정보를 제공합니다.
    - **3단계**: AI가 이미지에서 메뉴 이름과 가격을 정확히 읽어옵니다.
    - **4단계**: 추출된 결과를 표 형식으로 확인하고 틀린 부분이 있다면 직접 수정하세요.
    - **5단계**: 분류 대상인 대량의 음식 사진을 올리고 실행하면 자동으로 폴더링됩니다.
    """)
