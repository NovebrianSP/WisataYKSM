"""
Script untuk testing database functionality
"""
from app import DatabaseManager

def test_database():
    # Initialize database
    db = DatabaseManager("test_tourism.db")
    
    # Test user registration
    success, result = db.create_user(
        username="testuser",
        email="test@example.com", 
        password="password123",
        full_name="Test User",
        location="Jakarta, DKI Jakarta",
        age=25
    )
    
    if success:
        print(f"✅ User created successfully with ID: {result}")
        user_id = result
        
        # Test authentication
        auth_success, user_data = db.authenticate_user("testuser", "password123")
        if auth_success:
            print(f"✅ Authentication successful: {user_data}")
            
            # Test rating
            rating_success = db.add_user_rating(user_id, 1, 4.5)
            if rating_success:
                print("✅ Rating added successfully")
                
                # Test get ratings
                ratings = db.get_user_ratings(user_id)
                print(f"✅ User ratings: {ratings}")
                
                # Test preferences
                pref_success = db.update_user_preferences(user_id, "Budaya", 1.0)
                if pref_success:
                    print("✅ Preferences updated")
                    
                    preferences = db.get_user_preferences(user_id)
                    print(f"✅ User preferences: {preferences}")
                else:
                    print("❌ Failed to update preferences")
            else:
                print("❌ Failed to add rating")
        else:
            print(f"❌ Authentication failed: {user_data}")
    else:
        print(f"❌ User creation failed: {result}")

if __name__ == "__main__":
    test_database()
