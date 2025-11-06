import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Memuat dataset dari file CSV
    
    Args:
        filepath (str): Path ke file CSV
        
    Returns:
        pd.DataFrame: Dataset yang sudah dimuat
    """
    print(f"Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat! Shape: {df.shape}")
    return df


def handle_zero_values(df):
    """
    Menangani nilai 0 yang tidak wajar pada fitur medis
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset dengan nilai 0 yang sudah ditangani
    """
    print("\nHandling nilai 0 yang tidak wajar...")
    df_clean = df.copy()
    
    # Fitur yang tidak boleh bernilai 0
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_features:
        if col in df_clean.columns:
            # Hitung median dari nilai yang bukan 0
            median_value = df_clean[df_clean[col] != 0][col].median()
            # Replace 0 dengan median
            zeros_count = (df_clean[col] == 0).sum()
            df_clean[col] = df_clean[col].replace(0, median_value)
            print(f"  • {col}: {zeros_count} nilai 0 diganti dengan median ({median_value:.2f})")
    
    return df_clean


def remove_duplicates(df):
    """
    Menghapus data duplikat
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset tanpa duplikat
    """
    print("\nMenghapus data duplikat...")
    duplicates_before = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    duplicates_after = df_clean.duplicated().sum()
    
    print(f"  • Data duplikat dihapus: {duplicates_before}")
    print(f"  • Jumlah data sekarang: {len(df_clean)}")
    
    return df_clean


def remove_outliers(df, columns):
    """
    Menghapus outliers menggunakan IQR method
    
    Args:
        df (pd.DataFrame): Dataset input
        columns (list): List kolom yang akan diproses
        
    Returns:
        pd.DataFrame: Dataset tanpa outliers
    """
    print("\nMenghapus outliers menggunakan IQR method...")
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            after_count = len(df_clean)
            removed = before_count - after_count
            
            print(f"  • {col}: {removed} outliers dihapus")
    
    return df_clean


def feature_engineering(df):
    """
    Membuat fitur baru dari fitur yang ada
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset dengan fitur baru
    """
    print("\nFeature Engineering...")
    df_new = df.copy()
    
    # Membuat kategori BMI
    df_new['BMI_Category'] = pd.cut(df_new['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    print(f"  • Fitur BMI_Category dibuat")
    
    # Encoding kategori BMI
    df_new = pd.get_dummies(df_new, columns=['BMI_Category'], prefix='BMI')
    print(f"  • BMI_Category di-encode menjadi {len([c for c in df_new.columns if 'BMI_' in c])} kolom")
    
    return df_new


def standardize_features(df, target_column='Outcome'):
    """
    Standarisasi fitur menggunakan StandardScaler
    
    Args:
        df (pd.DataFrame): Dataset input
        target_column (str): Nama kolom target
        
    Returns:
        tuple: (X_scaled_df, y, scaler)
    """
    print("\nStandarisasi fitur...")
    
    # Split X dan y
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"  • Fitur berhasil distandarisasi")
    print(f"  • Shape X: {X_scaled_df.shape}")
    print(f"  • Shape y: {y.shape}")
    
    return X_scaled_df, y, scaler


def preprocess_pipeline(input_filepath, output_filepath=None):
    """
    Pipeline lengkap untuk preprocessing data
    
    Args:
        input_filepath (str): Path ke file CSV input
        output_filepath (str): Path untuk menyimpan hasil (optional)
        
    Returns:
        tuple: (X_scaled, y, df_final)
    """
    print("=" * 60)
    print("MEMULAI PIPELINE PREPROCESSING")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(input_filepath)
    
    # 2. Handle zero values
    df = handle_zero_values(df)
    
    # 3. Remove duplicates
    df = remove_duplicates(df)
    
    # 4. Remove outliers
    outlier_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df = remove_outliers(df, outlier_columns)
    
    # 5. Feature engineering
    df = feature_engineering(df)
    
    # 6. Standardize features
    X_scaled, y, scaler = standardize_features(df)
    
    # 7. Gabungkan untuk disimpan
    df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    
    # 8. Save hasil preprocessing
    if output_filepath:
        print(f"\nMenyimpan hasil preprocessing ke: {output_filepath}")
        df_final.to_csv(output_filepath, index=False)
        print(f"File berhasil disimpan!")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)
    print(f"Dataset akhir: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    
    return X_scaled, y, df_final, scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data menjadi training dan testing set
    
    Args:
        X: Features
        y: Target
        test_size (float): Proporsi data test
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n🔧 Split data train-test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  • Training set: {X_train.shape[0]} samples")
    print(f"  • Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Path file
    input_file = "./diabetes_raw.csv"
    output_file = "diabetes_preprocessing.csv"
    
    # Jalankan preprocessing pipeline
    X_scaled, y, df_final, scaler = preprocess_pipeline(input_file, output_file)
    
    # Split train-test
    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)
    
    print("\nScript selesai dijalankan!")
    print("File hasil: diabetes_preprocessing.csv")