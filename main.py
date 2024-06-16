import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def compute_eigen(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матриця має бути квадратною")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        A_v = np.dot(matrix, eigenvectors[:, i])
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        assert np.allclose(A_v, lambda_v), f"Некоректне власне значення {eigenvalues[i]}"

    return eigenvalues, eigenvectors


A = np.array([[1 , 1],
              [1, 1]])

eigenvalues, eigenvectors = compute_eigen(A)
print("Власні значення:", eigenvalues)
print("Власні вектори:\n", eigenvectors)


image_raw = imread('C:\\Users\\nekit\\Downloads\\1.jpeg')
image_shape = image_raw.shape
print("Розмір зображення (висота, ширина, канали):", image_shape)

plt.imshow(image_raw)
plt.title("Кольорове зображення")
plt.axis('off')
plt.show()

# Перетворення зображення в чорно-біле
image_sum = image_raw.sum(axis=2)
print("Розмір зображення (висота, ширина):", image_sum.shape)

# Нормалізація зображення
image_bw = image_sum / image_sum.max()
print("Максимальне значення чорно-білого зображення:", image_bw.max())

# Виведення чорно-білого зображення
plt.imshow(image_bw, cmap='gray')
plt.title("Чорно-біле зображення")
plt.axis('off')
plt.show()

# Застосування PCA
pca = PCA()
pca.fit(image_bw)

# Кумулятивна дисперсія
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Кількість компонент для покриття 95% дисперсії: {num_components_95}")

# Візуалізація cumulative variance
plt.plot(cumulative_variance)
plt.xlabel('Кількість компонентів')
plt.ylabel('Кумулятивна дисперсія')
plt.title('Кумулятивна дисперсія для PCA')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=num_components_95, color='r', linestyle='--')
plt.show()

# Реконструкція зображення
pca = PCA(n_components=num_components_95)
image_bw_pca = pca.fit_transform(image_bw)
image_bw_reconstructed = pca.inverse_transform(image_bw_pca)

plt.imshow(image_bw_reconstructed, cmap='gray')
plt.title(f'Реконструкція зображення з {num_components_95} компонентами')
plt.axis('off')
plt.show()

# Дослідження впливу різної кількості компонентів
components_to_test = [5, 15,25,75,100,170 ]

for n_components in components_to_test:
    pca = PCA(n_components=n_components)
    image_bw_pca = pca.fit_transform(image_bw)
    image_bw_reconstructed = pca.inverse_transform(image_bw_pca)

    plt.imshow(image_bw_reconstructed, cmap='gray')
    plt.title(f'Реконструкція з {n_components} компонентами')
    plt.axis('off')
    plt.show()


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    diagonalized_key_matrix_inv = np.dot(P, np.dot(np.linalg.inv(D), P_inv))
    decrypted_vector = np.dot(diagonalized_key_matrix_inv, encrypted_vector)

    decrypted_vector = np.real(decrypted_vector)
    decrypted_message = ''.join([chr(int(np.round(num))) for num in decrypted_vector])

    return decrypted_message


message = "Hello, World!"
key_matrix = np.random.randint(0,256,(len(message), len(message)))

encrypted_message = encrypt_message(message, key_matrix)
print("Original Message:", message)
print("Encrypted Message:", encrypted_message)

decrypted_message = decrypt_message(encrypted_message, key_matrix)
print("Decrypted Message:", decrypted_message)

