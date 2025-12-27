"""
데이터 파이프라인 자동화 스크립트
- HSI 계산 → 팀 템플릿 생성 → 검증
"""
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def run_script(script_path: str) -> bool:
    """Python 스크립트 실행"""
    logger.info(f"🔄 실행 중: {script_path}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✅ 완료: {script_path}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 오류 발생: {script_path}")
        logger.error(e.stderr)
        return False


def main():
    """전체 데이터 파이프라인 실행"""
    print("=" * 70)
    print("🚀 K-Scout HSI 데이터 파이프라인 시작")
    print("=" * 70)
    
    scripts = [
        ("1. HSI 계산", "src/hsi_calculator.py"),
        ("2. 팀 템플릿 생성", "src/team_profiler.py")
    ]
    
    for step, script in scripts:
        print(f"\n{'='*70}")
        print(f"📌 {step}")
        print("="*70)
        
        if not Path(script).exists():
            logger.error(f"❌ 파일을 찾을 수 없습니다: {script}")
            return False
        
        if not run_script(script):
            logger.error(f"❌ 파이프라인 중단: {step}에서 오류 발생")
            return False
    
    print("\n" + "="*70)
    print("✅ 전체 파이프라인 완료!")
    print("="*70)
    print("\n📊 생성된 파일:")
    print("  - output/hsi_scores_2024.csv")
    print("  - output/player_insights.json")
    print("  - output/team_templates.json")
    print("  - logs/hsi_pipeline.log")
    print("\n🚀 이제 Streamlit 앱을 실행하세요:")
    print("  streamlit run app.py")
    print("="*70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

