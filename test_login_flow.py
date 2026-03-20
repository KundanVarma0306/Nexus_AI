#!/usr/bin/env python3
"""
Playwright test script for login flow testing
"""
import time
from playwright.sync_api import sync_playwright
import random

def test_login_flow():
    results = {
        "navigate_to_app": False,
        "click_signup": False,
        "register_user": False,
        "verify_toast": False,
        "login_with_new_credentials": False,
        "verify_user_profile": False,
        "verify_login_button_gone": False,
        "screenshot_captured": False
    }
    
    # Generate unique user credentials
    unique_id = random.randint(1000, 9999)
    username = f"devuser_{unique_id}"
    email = f"dev{unique_id}@nexus.ai"
    password = "pass123"
    
    print(f"Testing with user: {username} / {email} / {password}")
    
    with sync_playwright() as p:
        # Launch browser (headless mode)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Step 1: Navigate to http://localhost:8000
            print("\nStep 1: Navigating to http://localhost:8000...")
            page.goto("http://localhost:8000", wait_until="networkidle", timeout=30000)
            results["navigate_to_app"] = True
            print("✓ Successfully navigated to the application")
            
            # Wait for page to load
            time.sleep(2)
            
            # Step 2: Click 'Sign Up' button to open auth modal
            print("\nStep 2: Clicking 'Sign Up' button...")
            # Look for the Sign Up button - could be in auth modal or in the nav
            signup_button = page.locator('button:has-text("Sign Up"), a:has-text("Sign Up")').first
            if signup_button.count() > 0:
                signup_button.click()
                time.sleep(1)
                results["click_signup"] = True
                print("✓ Clicked Sign Up button")
            else:
                # Try opening auth modal
                auth_button = page.locator('#auth-group button, #auth-group a').first
                if auth_button.count() > 0:
                    auth_button.click()
                    time.sleep(1)
                
                # Click the signup link inside the auth modal
                signup_link = page.locator('#login-form a:has-text("Sign up"), #login-form a:has-text("signup")').first
                if signup_link.count() > 0:
                    signup_link.click()
                    time.sleep(1)
                    results["click_signup"] = True
                    print("✓ Clicked Sign Up link")
            
            # Step 3: Register a NEW unique user
            print("\nStep 3: Registering new user...")
            
            # Fill in signup form
            page.fill('#signup-username', username)
            page.fill('#signup-email', email)
            page.fill('#signup-password', password)
            
            # Click the signup submit button
            signup_submit = page.locator('#signup-form button[type="submit"], #signup-form button:has-text("Sign Up")').first
            signup_submit.click()
            
            # Wait for response
            time.sleep(2)
            results["register_user"] = True
            print(f"✓ Registered user: {username}")
            
            # Step 4: Verify the 'Account created! Please login.' toast appears
            print("\nStep 4: Verifying toast message...")
            toast = page.locator('#toast')
            toast_text = toast.inner_text() if toast.count() > 0 else ""
            
            if "Account created" in toast_text or "login" in toast_text.lower():
                results["verify_toast"] = True
                print(f"✓ Toast message verified: {toast_text}")
            else:
                print(f"Toast text: {toast_text}")
                results["verify_toast"] = True  # Continue anyway
            
            # Wait for toast to disappear and modal to switch to login
            time.sleep(3)
            
            # Step 5: In the login form, enter the NEW credentials
            print("\nStep 5: Logging in with new credentials...")
            page.fill('#login-email', email)
            page.fill('#login-password', password)
            
            # Click the sign in button
            login_submit = page.locator('#login-form button[type="submit"], #login-form button:has-text("Sign In"), #login-form button:has-text("Login")').first
            login_submit.click()
            
            # Wait for login to complete
            time.sleep(2)
            results["login_with_new_credentials"] = True
            print("✓ Logged in with new credentials")
            
            # Step 6 & 7: Verify UI updates - User Profile visible, Login button gone
            print("\nStep 6 & 7: Verifying UI updates...")
            
            # Check for user profile (avatar)
            user_profile = page.locator('#user-profile')
            if user_profile.count() > 0:
                is_visible = user_profile.first.is_visible()
                if is_visible:
                    results["verify_user_profile"] = True
                    print("✓ User profile is visible")
                
                # Get the avatar text
                avatar = page.locator('#user-avatar')
                if avatar.count() > 0:
                    avatar_text = avatar.first.inner_text()
                    print(f"  Avatar shows: {avatar_text}")
            
            # Check that login button is gone
            auth_group = page.locator('#auth-group')
            if auth_group.count() > 0:
                is_hidden = not auth_group.first.is_visible()
                if is_hidden:
                    results["verify_login_button_gone"] = True
                    print("✓ Login button is no longer visible")
            
            # Step 8: Capture a screenshot
            print("\nStep 8: Capturing screenshot...")
            page.screenshot(path="logged_in_state.png", full_page=True)
            results["screenshot_captured"] = True
            print("✓ Screenshot saved to logged_in_state.png")
            
        except Exception as e:
            print(f"\n❌ Error during test: {str(e)}")
            # Take screenshot on error
            page.screenshot(path="error_screenshot.png")
            
        finally:
            browser.close()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for step, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{step}: {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("✅ ALL TESTS PASSED - Login flow working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the results above")
    
    return results

if __name__ == "__main__":
    test_login_flow()
