"""
EduShield - Data Preprocessing Module
Processes Open University Learning Analytics Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class StudentDataProcessor:
    """Process and engineer features from student data"""
    
    def __init__(self, data_path='/mnt/user-data/uploads'):
        self.data_path = Path(__file__).parent

        # self.data_path = Path(data_path)
        
    def load_data(self):
        """Load all CSV files"""
        print("📚 Loading datasets...")
        
        self.student_info = pd.read_csv(self.data_path / 'studentInfo.csv')
        self.assessments = pd.read_csv(self.data_path / 'assessments.csv')
        self.student_assessment = pd.read_csv(self.data_path / 'studentAssessment.csv')
        self.student_registration = pd.read_csv(self.data_path / 'studentRegistration.csv')
        self.courses = pd.read_csv(self.data_path / 'courses.csv')
        self.vle = pd.read_csv(self.data_path / 'vle.csv')
        
        print(f"✅ Loaded {len(self.student_info)} student records")
        print(f"✅ Loaded {len(self.student_assessment)} assessment records")
        print(f"✅ Loaded {len(self.student_registration)} registration records")
        
    def engineer_demographic_features(self, df):
        """Engineer demographic features"""
        print("\n👥 Engineering demographic features...")
        
        # Gender encoding
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        # Disability encoding
        df['disability_encoded'] = (df['disability'] == 'Y').astype(int)
        
        # Education level encoding
        education_mapping = {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        }
        df['education_level'] = df['highest_education'].map(education_mapping).fillna(2)
        
        # Age band encoding
        age_mapping = {
            '0-35': 1,
            '35-55': 2,
            '55<=': 3
        }
        df['age_encoded'] = df['age_band'].map(age_mapping).fillna(1)
        
        # IMD (Index of Multiple Deprivation) - extract numeric value
        df['imd_numeric'] = df['imd_band'].str.extract(r'(\d+)').astype(float)
        df['imd_numeric'] = df['imd_numeric'].fillna(50)  # Fill with median
        
        return df
    
    def engineer_academic_features(self, df):
        """Engineer academic background features"""
        print("📖 Engineering academic features...")
        
        # Previous attempts
        df['num_of_prev_attempts'] = df['num_of_prev_attempts'].fillna(0)
        df['has_previous_attempts'] = (df['num_of_prev_attempts'] > 0).astype(int)
        
        # Credits
        df['studied_credits'] = df['studied_credits'].fillna(df['studied_credits'].median())
        
        return df
    
    def engineer_assessment_features(self, df):
        """Engineer assessment performance features"""
        print("📝 Engineering assessment features...")
        
        # Merge assessment data
        assessment_stats = self.student_assessment.groupby('id_student').agg({
            'score': ['mean', 'std', 'min', 'max', 'count'],
            'date_submitted': ['mean', 'std'],
            'is_banked': 'sum'
        }).reset_index()
        
        assessment_stats.columns = ['id_student', 'avg_score', 'std_score', 'min_score', 
                                    'max_score', 'num_assessments', 'avg_submission_day', 
                                    'submission_consistency', 'banked_assessments']
        
        # Fill missing scores with course average
        assessment_stats['avg_score'] = assessment_stats['avg_score'].fillna(50)
        assessment_stats['std_score'] = assessment_stats['std_score'].fillna(0)
        assessment_stats['min_score'] = assessment_stats['min_score'].fillna(0)
        assessment_stats['max_score'] = assessment_stats['max_score'].fillna(0)
        assessment_stats['num_assessments'] = assessment_stats['num_assessments'].fillna(0)
        
        # Merge with main dataframe
        df = df.merge(assessment_stats, on='id_student', how='left')
        
        # Fill NaN for students with no assessments
        assessment_cols = ['avg_score', 'std_score', 'min_score', 'max_score', 
                          'num_assessments', 'avg_submission_day', 'submission_consistency']
        for col in assessment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Calculate score trend (improvement/decline)
        student_scores = self.student_assessment.sort_values(['id_student', 'date_submitted'])
        
        def calculate_trend(group):
            if len(group) < 2:
                return 0
            scores = group['score'].values
            # Calculate difference between second half and first half averages
            mid = len(scores) // 2
            if mid == 0:
                return 0
            first_half = np.mean(scores[:mid])
            second_half = np.mean(scores[mid:])
            return second_half - first_half
        
        trend = student_scores.groupby('id_student').apply(calculate_trend).reset_index()
        trend.columns = ['id_student', 'score_trend']
        df = df.merge(trend, on='id_student', how='left')
        df['score_trend'] = df['score_trend'].fillna(0)
        
        # Late submissions
        def count_late_submissions(group):
            # Assuming submissions after day 30 from assessment date are "late"
            return (group['date_submitted'] > 30).sum()
        
        late_subs = student_scores.groupby('id_student').apply(count_late_submissions).reset_index()
        late_subs.columns = ['id_student', 'late_submissions']
        df = df.merge(late_subs, on='id_student', how='left')
        df['late_submissions'] = df['late_submissions'].fillna(0)
        
        # First half vs second half performance
        def half_performance(group):
            if len(group) < 2:
                return pd.Series({'first_half_avg': 0, 'second_half_avg': 0})
            scores = group.sort_values('date_submitted')['score'].values
            mid = len(scores) // 2
            if mid == 0:
                return pd.Series({'first_half_avg': scores[0], 'second_half_avg': scores[0]})
            return pd.Series({
                'first_half_avg': np.mean(scores[:mid]),
                'second_half_avg': np.mean(scores[mid:])
            })
        
        half_perf = student_scores.groupby('id_student').apply(half_performance).reset_index()
        df = df.merge(half_perf, on='id_student', how='left')
        df['first_half_avg_score'] = df['first_half_avg'].fillna(0)
        df['second_half_avg_score'] = df['second_half_avg'].fillna(0)
        df = df.drop(['first_half_avg', 'second_half_avg'], axis=1, errors='ignore')
        
        return df
    
    def engineer_registration_features(self, df):
        """Engineer registration-related features"""
        print("📋 Engineering registration features...")
        
        # Merge registration data
        reg_data = self.student_registration[['id_student', 'date_registration', 'date_unregistration']].copy()
        
        # Registration timing
        reg_data['registration_day'] = reg_data['date_registration']
        reg_data['early_registration'] = (reg_data['date_registration'] < -30).astype(int)
        reg_data['late_registration'] = (reg_data['date_registration'] > 0).astype(int)
        reg_data['unregistered'] = reg_data['date_unregistration'].notna().astype(int)
        
        # Group by student (in case of multiple registrations, take first)
        reg_features = reg_data.groupby('id_student').agg({
            'registration_day': 'first',
            'early_registration': 'max',
            'late_registration': 'max',
            'unregistered': 'max'
        }).reset_index()
        
        df = df.merge(reg_features, on='id_student', how='left')
        
        # Fill missing registration data
        df['registration_day'] = df['registration_day'].fillna(0)
        df['early_registration'] = df['early_registration'].fillna(0)
        df['late_registration'] = df['late_registration'].fillna(0)
        df['unregistered'] = df['unregistered'].fillna(0)
        
        return df
    
    def create_target_variable(self, df):
        """Create target variable (dropout or not)"""
        print("🎯 Creating target variable...")
        
        # Target: 1 if Withdrawn or Fail, 0 if Pass or Distinction
        df['dropout'] = df['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
        
        print(f"✅ Dropout rate: {df['dropout'].mean()*100:.2f}%")
        print(f"   - Dropouts: {df['dropout'].sum()}")
        print(f"   - Retained: {(df['dropout'] == 0).sum()}")
        
        return df
    
    def process_all(self):
        """Main processing pipeline"""
        print("\n" + "="*60)
        print("🛡️  EDUSHIELD - DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Start with student info
        df = self.student_info.copy()
        
        # Engineer features
        df = self.engineer_demographic_features(df)
        df = self.engineer_academic_features(df)
        df = self.engineer_assessment_features(df)
        df = self.engineer_registration_features(df)
        df = self.create_target_variable(df)
        
        # Select final features
        feature_columns = [
            # Demographics (5)
            'gender_encoded', 'disability_encoded', 'education_level', 
            'age_encoded', 'imd_numeric',
            
            # Academic Background (3)
            'num_of_prev_attempts', 'has_previous_attempts', 'studied_credits',
            
            # Assessment Performance (11)
            'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
            'first_half_avg_score', 'second_half_avg_score', 'score_trend',
            'submission_consistency', 'late_submissions', 'avg_submission_day',
            
            # Registration (4)
            'registration_day', 'early_registration', 'late_registration', 'unregistered',
            
            # Target
            'dropout'
        ]
        
        # Keep identification columns
        id_columns = ['id_student', 'code_module', 'code_presentation', 'gender', 
                     'region', 'highest_education', 'age_band', 'disability', 'final_result']
        
        # Create final dataset
        final_df = df[id_columns + feature_columns].copy()
        
        # Remove rows with missing target
        final_df = final_df.dropna(subset=['dropout'])
        
        print(f"\n📊 Final dataset shape: {final_df.shape}")
        print(f"   - Total features: {len(feature_columns) - 1}")  # Excluding target
        print(f"   - Total samples: {len(final_df)}")
        
        # Save processed data
        output_path = Path(__file__).parent / "data" / "processed_student_data.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n💾 Processed data saved to: {output_path}")
        
        # Display feature summary
        print("\n" + "="*60)
        print("📈 FEATURE SUMMARY")
        print("="*60)
        
        feature_groups = {
            'Demographics': ['gender_encoded', 'disability_encoded', 'education_level', 'age_encoded', 'imd_numeric'],
            'Academic': ['num_of_prev_attempts', 'has_previous_attempts', 'studied_credits'],
            'Assessment': ['avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
                          'first_half_avg_score', 'second_half_avg_score', 'score_trend',
                          'submission_consistency', 'late_submissions', 'avg_submission_day'],
            'Registration': ['registration_day', 'early_registration', 'late_registration', 'unregistered']
        }
        
        for group_name, features in feature_groups.items():
            print(f"\n{group_name} Features ({len(features)}):")
            for feat in features:
                if feat in final_df.columns:
                    print(f"  ✓ {feat}")
        
        print("\n" + "="*60)
        print("✅ DATA PREPROCESSING COMPLETE!")
        print("="*60)
        
        return final_df


if __name__ == "__main__":
    processor = StudentDataProcessor()
    df = processor.process_all()
    
    # Display sample
    print("\n📋 Sample processed data:")
    print(df.head(3))