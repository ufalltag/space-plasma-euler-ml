"""
Космическая плазма — задание.

Численное решение связанной системы уравнений для квадратов магнитного и
электрического полей H^2(r), E^2(r) в плазменном цилиндре методом Эйлера,
проверка порядка численной схемы, генерация датасета и обучение четырёх
моделей машинного обучения (линейная регрессия, перцептрон с tanh, NN ReLU,
NN ELU) — по одной на каждый из вариантов i%4 = 0,1,2,3.

Физическая постановка (см. фотографии доски):

    (1/r) d/dr ( (r/sigma) dH^2/dr ) = 2 sigma E^2
    (1/r) d/dr ( (1/r)    d(r^2 E^2)/dr ) = 2 mu0 omega H^2

    BC: dH^2/dr(0) = 0,  H^2(R) = H_R^2
        E^2(0)     = 0,  d(r^2 E^2)/dr|_0 = 0

Подстановка:
    v = (r/sigma) dH^2/dr,  y = r^2 E^2,  w = (1/r) dy/dr

Получаем систему 4-х ОДУ 1-го порядка:

    dH^2/dr = (sigma / r) v
    dv/dr   = 2 r sigma E^2 = 2 sigma y / r
    dy/dr   = r w
    dw/dr   = 2 r mu0 omega H^2

В нуле все правые части обращаются в 0, особенностей нет.

Граничная задача с условием H^2(R)=H_R^2 решается простым масштабированием —
система линейна и однородна по (H^2, v, y, w), стартует из (H0, 0, 0, 0).
Поэтому: при H0_unit=1 интегрируем один раз, получаем H^2(R)_unit, и
истинное H0 = H_R^2 / H^2(R)_unit. Все остальные величины масштабируются
тем же коэффициентом alpha.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf

mp.dps = 40
np.random.seed(42)

mu0 = 4 * np.pi * 1e-7  # магнитная постоянная, Гн/м

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 1. Метод Эйлера для системы плазменных уравнений
# =============================================================================

def euler_solve(sigma, R, omega, H0_sq=1.0, N=2000):
    """Прямое интегрирование системы методом Эйлера от r=0 до r=R.

    Состояние: (H^2, v, y, w), где:
        v = (r/sigma) dH^2/dr   =>   dH^2/dr = (sigma/r) v
        y = r^2 E^2
        w = (1/r) dy/dr

    Уравнения:
        dH^2/dr = (sigma/r) v          ( -> 0 при r->0 )
        dv/dr   = 2 r sigma E^2 = 2 sigma y / r
        dy/dr   = r w
        dw/dr   = 2 r mu0 omega H^2

    IC: H^2(0)=H0_sq, v(0)=0, y(0)=0, w(0)=0.
    """
    h = R / N
    r = np.linspace(0.0, R, N + 1)

    H_sq = np.zeros(N + 1)
    v    = np.zeros(N + 1)
    y    = np.zeros(N + 1)
    w    = np.zeros(N + 1)

    H_sq[0] = H0_sq

    for i in range(N):
        ri = r[i]
        if ri > 0.0:
            dH_sq = (sigma / ri) * v[i]
            E_sq_i = y[i] / (ri * ri)
            dv = 2.0 * sigma * y[i] / ri  # = 2 r sigma E^2 1 уравнение
        else:
            dH_sq = 0.0
            E_sq_i = 0.0
            dv = 0.0

        dy = ri * w[i]
        dw = 2.0 * ri * mu0 * omega * H_sq[i] # 2 уравнение

        H_sq[i + 1] = H_sq[i] + h * dH_sq
        v[i + 1]    = v[i]    + h * dv
        y[i + 1]    = y[i]    + h * dy
        w[i + 1]    = w[i]    + h * dw

    E_sq = np.zeros_like(r)
    E_sq[1:] = y[1:] / (r[1:] ** 2)
    return r, H_sq, E_sq, y, w


def solve_bvp(H_R, sigma, R, omega, N=1000):
    """Граничная задача через масштабирование линейного решения."""
    r, H_sq_u, E_sq_u, y_u, w_u = euler_solve(sigma, R, omega, H0_sq=1.0, N=N)
    H_sq_R_unit = H_sq_u[-1]
    if H_sq_R_unit <= 0:
        H_sq_R_unit = 1.0
    alpha = (H_R ** 2) / H_sq_R_unit
    H_sq = alpha * H_sq_u
    E_sq = alpha * E_sq_u
    H_0 = np.sqrt(max(H_sq[0], 0.0))
    E_R = np.sqrt(max(E_sq[-1], 0.0))
    return r, H_sq, E_sq, H_0, E_R


# =============================================================================
# 2. Проверка порядка численной схемы (метод Эйлера → 1-й порядок)
# =============================================================================
# Берём упрощённый случай: H^2(r) = H0^2 = const (не реагирует на E^2).
# Тогда:
#     w(r) = mu0 omega H0^2 r^2
#     y(r) = mu0 omega H0^2 r^4 / 4
#     E^2(r) = mu0 omega H0^2 r^2 / 4
# Это даёт точное аналитическое решение, на котором проверяется порядок схемы.

def euler_simple(omega, R, H0_sq, N):
    """Эйлер для уравнения E^2 при заданном постоянном H^2 = H0_sq."""
    h = R / N
    r = np.linspace(0.0, R, N + 1)
    y = np.zeros(N + 1)
    w = np.zeros(N + 1)
    for i in range(N):
        y[i + 1] = y[i] + h * r[i] * w[i]
        w[i + 1] = w[i] + h * 2.0 * r[i] * mu0 * omega * H0_sq
    return r, y, w


def verify_order(omega=2 * np.pi * 1.76, R=1.0, H0_sq=1.0):
    """Считает максимальную ошибку Эйлера для разных h, оценивает порядок."""
    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    hs, errs_y, errs_w = [], [], []
    for N in Ns:
        r, y_num, w_num = euler_simple(omega, R, H0_sq, N)
        y_ex = mu0 * omega * H0_sq * r ** 4 / 4.0
        w_ex = mu0 * omega * H0_sq * r ** 2
        err_y = np.max(np.abs(y_num - y_ex))
        err_w = np.max(np.abs(w_num - w_ex))
        hs.append(R / N)
        errs_y.append(err_y)
        errs_w.append(err_w)

    hs = np.array(hs)
    errs_y = np.array(errs_y)
    errs_w = np.array(errs_w)

    # МНК-оценка наклона log(err) ~ p log(h) + const
    p_y = np.polyfit(np.log(hs), np.log(errs_y), 1)[0]
    p_w = np.polyfit(np.log(hs), np.log(errs_w), 1)[0]
    return hs, errs_y, errs_w, p_y, p_w


# =============================================================================
# 3. ML — реализация моделей на чистом numpy
# =============================================================================

def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


class LinearRegressor:
    """Линейная регрессия (нормальное уравнение)."""
    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        # Регуляризация для устойчивости
        A = Xb.T @ Xb + 1e-8 * np.eye(Xb.shape[1])
        self.theta = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return Xb @ self.theta


def _act(name):
    if name == 'tanh':
        return (lambda z: np.tanh(z),
                lambda z, a: 1.0 - a * a)
    if name == 'relu':
        return (lambda z: np.maximum(0.0, z),
                lambda z, a: (z > 0).astype(float))
    if name == 'elu':
        alpha = 1.0
        return (lambda z: np.where(z > 0, z, alpha * (np.exp(np.minimum(z, 0)) - 1)),
                lambda z, a: np.where(z > 0, 1.0, a + alpha))
    raise ValueError(name)


class NeuralNet:
    """Многослойный перцептрон с произвольной активацией.

    Обучение методом mini-batch Adam с MSE-потерей.
    """
    def __init__(self, sizes, activation='tanh', lr=1e-2, epochs=2000,
                 batch_size=64, seed=0):
        self.sizes = sizes
        self.act_name = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _init(self):
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            # Glorot/He
            scale = np.sqrt(2.0 / n_in)
            self.W.append(self.rng.normal(0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _forward(self, X):
        f, df = _act(self.act_name)
        zs, acts = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            if i < len(self.W) - 1:
                a = f(z)
            else:
                a = z  # линейный выход
            acts.append(a)
        return zs, acts

    def predict(self, X):
        return self._forward(X)[1][-1].ravel() # Forward pass — получить предсказание

    def fit(self, X, y):
        self._init()
        f, df = _act(self.act_name)
        n = X.shape[0]
        y = y.reshape(-1, 1)
        loss_curve = []
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        for ep in range(self.epochs):
            idx = self.rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch_size):
                ib = idx[s:s + self.batch_size]
                Xb, yb = X[ib], y[ib]
                zs, acts = self._forward(Xb)
                # backprop посчитать градиенты
                dz = (acts[-1] - yb) * 2.0 / Xb.shape[0]  # MSE grad Loss function — то, что модель минимизирует во время обучения
                grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
                for i in reversed(range(len(self.W))):
                    a_prev = acts[i]
                    grads_W[i] = a_prev.T @ dz
                    grads_b[i] = dz.sum(axis=0)
                    if i > 0:
                        da = dz @ self.W[i].T
                        dz = da * df(zs[i - 1], acts[i])
                # Adam update
                t += 1
                lr_t = self.lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                for i in range(len(self.W)):
                    self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
                    self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * grads_W[i] ** 2
                    self.W[i] -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + eps) # Обновить
                    self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
                    self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * grads_b[i] ** 2
                    self.b[i] -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + eps)
                ep_loss += float(((acts[-1] - yb) ** 2).sum())
            loss_curve.append(ep_loss / n)
        self.loss_curve_ = loss_curve
        return self


class TanhPerceptron(NeuralNet):
    """Один скрытый слой, активация tanh — «перцептрон на tg»."""
    def __init__(self, n_features, hidden=24, **kw):
        super().__init__([n_features, hidden, 1], activation='tanh',
                         lr=5e-3, epochs=600, batch_size=64, **kw)


class ReLUNet(NeuralNet):
    def __init__(self, n_features, hidden=(32, 16), **kw):
        super().__init__([n_features, *hidden, 1], activation='relu',
                         lr=5e-3, epochs=800, batch_size=64, **kw)


class ELUNet(NeuralNet):
    def __init__(self, n_features, hidden=(32, 16), **kw):
        super().__init__([n_features, *hidden, 1], activation='elu',
                         lr=5e-3, epochs=800, batch_size=64, **kw)


# =============================================================================
# 4. Генерация датасета
# =============================================================================
# 4 признака:    H_R [А/м], sigma [См/м], R [м], omega [рад/с]
# Цель:          E(R) — для чётного i,  H(0) — для нечётного i
# Сетка:         8x8x8x2 (или перестановка), всего ровно 1024 строки.

def make_grids(i_var):
    """Возвращает 4 одномерных массива значений признаков для варианта i.

    Частота для варианта i: f_i = 1.76 * i  ГГц (если i=0, берём f=1.76 ГГц).
    Сетка omega — вокруг f_i (от 0.5 f_i до 2.0 f_i).
    Размер «сжатой» оси (2 значения вместо 8) определяется case = i_var % 4.
    """
    f_i = 1.76 * max(i_var, 1)                           # ГГц
    H_R_vals  = np.linspace(100.0, 1000.0, 8)            # амплитуда поля на границе
    sig_vals  = np.logspace(-2.0, 1.0, 8)                # проводимость
    R_vals    = np.linspace(0.10, 1.00, 8)               # радиус столба
    f_vals    = np.linspace(0.5, 2.0, 8) * f_i           # частоты, ГГц
    omega_vals = 2.0 * np.pi * f_vals * 1e9              # рад/с

    # Какой признак сжат до 2 значений
    case = i_var % 4
    if case == 0:
        omega_vals = np.array([omega_vals[0], omega_vals[-1]])
    elif case == 1:
        R_vals = np.array([R_vals[0], R_vals[-1]])
    elif case == 2:
        sig_vals = np.array([sig_vals[0], sig_vals[-1]])
    elif case == 3:
        H_R_vals = np.array([H_R_vals[0], H_R_vals[-1]])
    return H_R_vals, sig_vals, R_vals, omega_vals


def generate_dataset(i_var, N_euler=400):
    H_R_vals, sig_vals, R_vals, omega_vals = make_grids(i_var)
    rows = []
    for H_R in H_R_vals:
        for sig in sig_vals:
            for R in R_vals:
                for om in omega_vals:
                    _, _, _, H0, ER = solve_bvp(H_R, sig, R, om, N=N_euler)
                    rows.append([H_R, sig, R, om, H0, ER])
    arr = np.array(rows)
    X = arr[:, :4]
    H0 = arr[:, 4]
    ER = arr[:, 5]
    return X, H0, ER


# =============================================================================
# 5. Главный пайплайн: для каждого case прогоняем расчёт и обучение
# =============================================================================

def train_and_eval(i_var, X, y, target_name, feature_names, force_case=None):
    """Обучает модель на (X,y); force_case переопределяет выбор модели."""
    case = force_case if force_case is not None else i_var % 4
    # Простой train/test split (80/20)
    rng = np.random.default_rng(42 + i_var)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_te = n // 5
    te = idx[:n_te]; tr = idx[n_te:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    # Стандартизация
    Xs_tr, mu_x, sd_x = standardize(X_tr)
    Xs_te = (X_te - mu_x) / sd_x
    mu_y, sd_y = y_tr.mean(), y_tr.std() if y_tr.std() > 1e-12 else 1.0
    ys_tr = (y_tr - mu_y) / sd_y

    if case == 0:
        model = LinearRegressor()
        model_name = 'Линейная регрессия'
    elif case == 1:
        model = TanhPerceptron(n_features=X.shape[1])
        model_name = 'Перцептрон, активация tanh'
    elif case == 2:
        model = ReLUNet(n_features=X.shape[1])
        model_name = 'Нейросеть, активация ReLU'
    elif case == 3:
        model = ELUNet(n_features=X.shape[1])
        model_name = 'Нейросеть, активация ELU'

    model.fit(Xs_tr, ys_tr)
    pred_tr = model.predict(Xs_tr) * sd_y + mu_y
    pred_te = model.predict(Xs_te) * sd_y + mu_y

    mse_tr = float(np.mean((pred_tr - y_tr) ** 2)) # метрика
    mse_te = float(np.mean((pred_te - y_te) ** 2))
    rmse_tr = float(np.sqrt(mse_tr))
    rmse_te = float(np.sqrt(mse_te))
    # Относительная RMSE — % от средней цели
    rel_rmse = 100.0 * rmse_te / (np.mean(np.abs(y_te)) + 1e-30)

    loss_curve = getattr(model, 'loss_curve_', None)

    return {
        'i_var': i_var,
        'case': case,
        'model_name': model_name,
        'target_name': target_name,
        'feature_names': feature_names,
        'X_tr': X_tr, 'X_te': X_te,
        'y_tr': y_tr, 'y_te': y_te,
        'pred_tr': pred_tr, 'pred_te': pred_te,
        'mse_tr': mse_tr, 'mse_te': mse_te,
        'rmse_tr': rmse_tr, 'rmse_te': rmse_te,
        'rel_rmse_te_pct': rel_rmse,
        'loss_curve': loss_curve,
    }


def plot_case(res, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"i = {res['i_var']} (i%4 = {res['case']}) — "
                 f"{res['model_name']} → {res['target_name']}",
                 fontsize=13)

    # 1) кривая обучения
    ax = axes[0]
    if res['loss_curve'] is not None:
        ax.plot(res['loss_curve'], lw=1.2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('MSE (масштабир.)')
        ax.set_title('Кривая обучения')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'Закрытое решение\n(нормальное уравнение)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Кривая обучения')
        ax.axis('off')
    ax.grid(True, ls='--', alpha=0.5)

    # 2) Предсказание vs истина
    ax = axes[1]
    ax.scatter(res['y_tr'], res['pred_tr'], s=8, alpha=0.5, label='train')
    ax.scatter(res['y_te'], res['pred_te'], s=12, color='C3', label='test')
    lim = [min(res['y_tr'].min(), res['pred_tr'].min()),
           max(res['y_tr'].max(), res['pred_tr'].max())]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel(f"истинное {res['target_name']}")
    ax.set_ylabel('предсказание')
    ax.set_title(f"MSE_test = {res['mse_te']:.3e}")
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    # 3) Зависимость целевой переменной от наиболее важного признака
    ax = axes[2]
    fi = int(np.argmax(np.var(res['X_tr'], axis=0) > 0))  # любой переменный
    # Возьмём H_R (первый признак) — обычно самый влиятельный
    j = 0
    order = np.argsort(res['X_te'][:, j])
    ax.scatter(res['X_te'][order, j], res['y_te'][order], s=14, label='истина')
    ax.scatter(res['X_te'][order, j], res['pred_te'][order], s=14, marker='x',
               color='C3', label='модель')
    ax.set_xlabel(res['feature_names'][j])
    ax.set_ylabel(res['target_name'])
    ax.set_title('Срез: цель vs ' + res['feature_names'][j])
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)


def plot_model_comparison(results, savepath):
    """Сравнительный график для нескольких моделей на одном датасете (предсказание vs истина)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    i_var = results[0]['i_var']
    fig.suptitle(f'Сравнение моделей — i={i_var}, цель: {results[0]["target_name"]}', fontsize=13)
    for ax, res in zip(axes, results):
        ax.scatter(res['y_tr'], res['pred_tr'], s=8, alpha=0.4, label='train')
        ax.scatter(res['y_te'], res['pred_te'], s=12, color='C3', label='test')
        lim = [min(res['y_tr'].min(), res['pred_tr'].min()),
               max(res['y_tr'].max(), res['pred_tr'].max())]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_title(f"{res['model_name']}\nRMSE={res['rmse_te']:.3e}  ({res['rel_rmse_te_pct']:.1f}%)")
        ax.set_xlabel(f"истинное {res['target_name']}")
        ax.set_ylabel('предсказание')
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)


def plot_sample_solution(i_var, savepath):
    """Показать пример численного решения H^2(r), E^2(r) и H(r), E(r)."""
    H_R = 500.0
    sigma = 0.1
    R = 0.5
    f = 1.76e9 * max(i_var, 1)
    omega = 2 * np.pi * f
    r, H_sq, E_sq, H_0, E_R = solve_bvp(H_R, sigma, R, omega, N=2000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f'Пример численного решения (i={i_var}, i%4={i_var%4}, '
                 f'H_R={H_R} А/м, σ={sigma} См/м, R={R} м, f={f:.2e} Гц)',
                 fontsize=11)

    axes[0].plot(r, np.sqrt(np.maximum(H_sq, 0)), 'b-', lw=2)
    axes[0].axhline(H_R, color='gray', ls='--', label='H_R')
    axes[0].set_xlabel('r, м')
    axes[0].set_ylabel('|H(r)|, А/м')
    axes[0].set_title(f'H(0) = {H_0:.2f}')
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)

    axes[1].plot(r, np.sqrt(np.maximum(E_sq, 0)), 'r-', lw=2)
    axes[1].set_xlabel('r, м')
    axes[1].set_ylabel('|E(r)|, В/м')
    axes[1].set_title(f'E(R) = {E_R:.3e}')
    axes[1].grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)


# =============================================================================
# 7. Бенчмарк: качество модели vs размер датасета
# =============================================================================

def generate_dataset_random(i_var, n_samples, N_euler=200, seed=0):
    """Случайный датасет из n_samples точек равномерно из пространства параметров."""
    rng = np.random.default_rng(seed)
    f_i = 1.76 * max(i_var, 1)
    H_R   = rng.uniform(100.0, 1000.0, n_samples)
    sig   = np.exp(rng.uniform(np.log(1e-2), np.log(10.0), n_samples))
    R     = rng.uniform(0.10, 1.00, n_samples)
    f     = rng.uniform(0.5, 2.0, n_samples) * f_i
    omega = 2.0 * np.pi * f * 1e9
    rows = []
    for h, s, r, om in zip(H_R, sig, R, omega):
        _, _, _, H0, ER = solve_bvp(h, s, r, om, N=N_euler)
        rows.append([h, s, r, om, H0, ER])
    arr = np.array(rows)
    return arr[:, :4], arr[:, 4], arr[:, 5]


def _build_model(i_var):
    case = i_var % 4
    if case == 0:
        return LinearRegressor(), 'Линейная регрессия'
    elif case == 1:
        return TanhPerceptron(n_features=4), 'Перцептрон (tanh)'
    elif case == 2:
        return ReLUNet(n_features=4), 'Нейросеть (ReLU)'
    else:
        return ELUNet(n_features=4), 'Нейросеть (ELU)'


def _build_model_by_case(case):
    if case == 0:
        return LinearRegressor(), 'Линейная регрессия'
    elif case == 1:
        return TanhPerceptron(n_features=4), 'Перцептрон (tanh)'
    elif case == 2:
        return ReLUNet(n_features=4), 'Нейросеть (ReLU)'
    else:
        return ELUNet(n_features=4), 'Нейросеть (ELU)'


def dataset_size_benchmark_all_models(i_var, sizes=None, N_euler=150):
    """Бенчмарк всех 4 моделей на датасете одного варианта i.

    Тестовая выборка фиксирована (400 точек, seed=999).
    Обучающий пул — max(sizes) точек (seed=0).
    Возвращает dict: {case: [{n, rmse, rel_pct, model_name}, ...]}
    """
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024, 2048]

    even = (i_var % 2 == 0)
    n_test = 400

    print(f'  Генерация тестовой выборки (i={i_var}, {n_test} точек) ...')
    X_te, H0_te, ER_te = generate_dataset_random(i_var, n_test, N_euler, seed=999)
    y_te = ER_te if even else H0_te

    max_n = max(sizes)
    print(f'  Генерация пула ({max_n} точек) ...')
    X_pool, H0_pool, ER_pool = generate_dataset_random(i_var, max_n, N_euler, seed=0)
    y_pool = ER_pool if even else H0_pool

    all_results = {}
    for case in range(4):
        _, model_name = _build_model_by_case(case)
        print(f'  case={case} ({model_name}):')
        size_res = []
        for n in sizes:
            X_tr = X_pool[:n]
            y_tr = y_pool[:n]

            Xs_tr, mu_x, sd_x = standardize(X_tr)
            Xs_te_s = (X_te - mu_x) / sd_x
            mu_y = y_tr.mean()
            sd_y = max(float(y_tr.std()), 1e-12)
            ys_tr = (y_tr - mu_y) / sd_y

            model, _ = _build_model_by_case(case)
            model.fit(Xs_tr, ys_tr)
            pred = model.predict(Xs_te_s) * sd_y + mu_y

            rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
            rel  = 100.0 * rmse / (np.mean(np.abs(y_te)) + 1e-30)
            size_res.append({'n': n, 'rmse': rmse, 'rel_pct': rel, 'model_name': model_name})
            print(f'    n={n:5d}: RMSE={rmse:.4e}  rel={rel:.2f}%')

        all_results[case] = size_res
    return all_results


def plot_dataset_size_benchmark_all_models(i_var, all_results, savepath):
    """Два графика: RMSE и отн. RMSE vs N для всех 4 моделей на датасете i_var."""
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Все модели на датасете i={i_var} — качество vs размер выборки', fontsize=13)

    for case, size_res in all_results.items():
        ns   = [r['n']         for r in size_res]
        rmse = [r['rmse']      for r in size_res]
        rel  = [r['rel_pct']   for r in size_res]
        lbl  = size_res[0]['model_name']
        axes[0].plot(ns, rmse, 'o-', color=colors[case], label=lbl)
        axes[1].plot(ns, rel,  'o-', color=colors[case], label=lbl)

    for ax, ylabel, title in [
        (axes[0], 'RMSE (тестовая выборка)', 'RMSE vs N'),
        (axes[1], 'Отн. RMSE, %',            'Отн. RMSE (%) vs N'),
    ]:
        ax.set_xlabel('Размер обучающей выборки N')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)


def dataset_size_benchmark(sizes=None, i_vars=None, N_euler=150):
    """Для каждого варианта обучает модель на обучающих выборках разного размера.

    Тестовая выборка фиксирована (400 точек, seed=999), чтобы RMSE было сопоставимым.
    Обучающий пул — max(sizes) точек (seed=0), из которого берутся первые n строк.
    """
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024, 2048]
    if i_vars is None:
        i_vars = [0, 1, 2, 3]

    n_test = 400
    all_results = {}
    for i_var in i_vars:
        print(f'  Бенчмарк i={i_var}  (i%4={i_var % 4}): генерация данных ...')
        even = (i_var % 2 == 0)

        X_te, H0_te, ER_te = generate_dataset_random(i_var, n_test, N_euler, seed=999)
        y_te = ER_te if even else H0_te

        max_n = max(sizes)
        X_pool, H0_pool, ER_pool = generate_dataset_random(i_var, max_n, N_euler, seed=0)
        y_pool = ER_pool if even else H0_pool

        size_res = []
        for n in sizes:
            X_tr = X_pool[:n]
            y_tr = y_pool[:n]

            Xs_tr, mu_x, sd_x = standardize(X_tr)
            Xs_te_s = (X_te - mu_x) / sd_x
            mu_y = y_tr.mean()
            sd_y = max(float(y_tr.std()), 1e-12)
            ys_tr = (y_tr - mu_y) / sd_y

            model, _ = _build_model(i_var)
            model.fit(Xs_tr, ys_tr)
            pred = model.predict(Xs_te_s) * sd_y + mu_y

            rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
            rel  = 100.0 * rmse / (np.mean(np.abs(y_te)) + 1e-30)
            size_res.append({'n': n, 'rmse': rmse, 'rel_pct': rel})
            print(f'    n={n:5d}: RMSE={rmse:.4e}  rel={rel:.2f}%')

        all_results[i_var] = size_res
    return all_results


def plot_dataset_size_benchmark(all_results, savepath):
    model_labels = {
        0: 'Лин. регрессия (i=0)',
        1: 'Перцептрон tanh (i=1)',
        2: 'Нейросеть ReLU (i=2)',
        3: 'Нейросеть ELU  (i=3)',
        16: 'Лин. регрессия (i=16)',
    }
    colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3', 16: 'C4'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Качество модели vs размер обучающей выборки', fontsize=13)

    for i_var, size_res in all_results.items():
        ns   = [r['n']       for r in size_res]
        rmse = [r['rmse']    for r in size_res]
        rel  = [r['rel_pct'] for r in size_res]
        c = colors.get(i_var, f'C{i_var % 10}')
        lbl = model_labels.get(i_var, f'i={i_var} (i%4={i_var%4})')
        axes[0].plot(ns, rmse, 'o-', color=c, label=lbl)
        axes[1].plot(ns, rel,  'o-', color=c, label=lbl)

    for ax, ylabel, title in [
        (axes[0], 'RMSE (тестовая выборка)', 'RMSE vs N'),
        (axes[1], 'Отн. RMSE, %',            'Отн. RMSE (%) vs N'),
    ]:
        ax.set_xlabel('Размер обучающей выборки N')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)
    return fig


def plot_dataset_size_benchmark_detail(all_results, savepath):
    """Субграфики — по одному на каждый вариант i, отн. RMSE vs N."""
    model_names = {
        0: 'Линейная регрессия (i=0)',
        1: 'Перцептрон (tanh, i=1)',
        2: 'Нейросеть (ReLU, i=2)',
        3: 'Нейросеть (ELU, i=3)',
        16: 'Линейная регрессия (i=16)',
    }
    n_plots = len(all_results)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes_flat = axes.flat if nrows > 1 else list(axes)
    fig.suptitle('Кривые обучения (отн. RMSE % vs N) по вариантам', fontsize=13)

    for idx, (i_var, size_res) in enumerate(all_results.items()):
        ax = axes_flat[idx]
        ns  = [r['n']       for r in size_res]
        rel = [r['rel_pct'] for r in size_res]
        ax.plot(ns, rel, 'o-', color=f'C{idx}', lw=2)
        ax.set_xlabel('N (размер обучающей выборки)')
        ax.set_ylabel('Отн. RMSE, %')
        ax.set_title(model_names.get(i_var, f'i={i_var} (i%4={i_var%4})'))
        ax.set_xscale('log')
        ax.grid(True, ls='--', alpha=0.5)
        ax.axhline(rel[-1], ls=':', color='gray', alpha=0.7,
                   label=f'при N={ns[-1]}: {rel[-1]:.1f}%')
        ax.legend(fontsize=8)

    for idx in range(n_plots, nrows * ncols):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)


# =============================================================================
# 6. Точка входа
# =============================================================================

def main():
    print('=' * 70)
    print('Космическая плазма — задание')
    print('=' * 70)

    feature_names = ['H_R, А/м', 'σ, См/м', 'R, м', 'ω, рад/с']

    # ---- 6.1  Проверка порядка численной схемы Эйлера
    print('\n[1] Проверка порядка численной схемы')
    print('-' * 70)
    hs, eY, eW, pY, pW = verify_order()
    print(f'  Оценённый порядок по y: p ≈ {pY:.3f}  (теория: 1)')
    print(f'  Оценённый порядок по w: p ≈ {pW:.3f}  (теория: 1)')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Проверка порядка точности метода Эйлера', fontsize=13)
    axes[0].loglog(hs, eY, 'o-', label=f'err y, наклон ≈ {pY:.2f}')
    axes[0].loglog(hs, eW, 's-', label=f'err w, наклон ≈ {pW:.2f}')
    axes[0].loglog(hs, hs, 'k--', label='∼ h¹ (1-й порядок)')
    axes[0].set_xlabel('h')
    axes[0].set_ylabel('max ошибка')
    axes[0].set_title('e(h) в log-log')
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    axes[0].legend()

    axes[1].semilogx(hs, eY / hs, 'o-', label='err_y / h')
    axes[1].semilogx(hs, eW / hs, 's-', label='err_w / h')
    axes[1].set_xlabel('h')
    axes[1].set_ylabel('e / h  (≈ const если порядок 1)')
    axes[1].set_title('Отношение ошибки к h')
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'order_check.png'), dpi=140)
    plt.close(fig)

    # ---- 6.2  Для каждого варианта i — генерация данных, обучение, графики
    # Базовые i = 0..3 (по одному на каждый case = i%4) + конкретный i = 16.
    summary = []
    i_list = [0, 1, 2, 3, 16]
    for i_var in i_list:
        case = i_var % 4
        print('\n' + '-' * 70)
        print(f'[2.{i_var}] Вариант i = {i_var}  (i%4 = {case})')
        print('-' * 70)

        # Целевая величина
        if i_var % 2 == 0:
            target_name = 'E(R), В/м'
        else:
            target_name = 'H(0), А/м'

        print('  Генерация датасета (1024 строк) ...')
        X, H0, ER = generate_dataset(i_var, N_euler=300)
        y = ER if (i_var % 2 == 0) else H0
        print(f'  X.shape = {X.shape},   y.shape = {y.shape}')
        print(f'  y: min={y.min():.3e}  max={y.max():.3e}  mean={y.mean():.3e}')

        print('  Обучение модели ...')
        res = train_and_eval(i_var, X, y, target_name, feature_names)
        print(f'  Модель: {res["model_name"]}')
        print(f'  MSE  train = {res["mse_tr"]:.4e}')
        print(f'  MSE  test  = {res["mse_te"]:.4e}')
        print(f'  RMSE test  = {res["rmse_te"]:.4e}'
              f'   ({res["rel_rmse_te_pct"]:.2f}% от средней цели)')

        plot_case(res, os.path.join(FIG_DIR, f'case_i{i_var}.png'))
        plot_sample_solution(i_var, os.path.join(FIG_DIR, f'sample_solution_i{i_var}.png'))

        summary.append({
            'i_var': i_var,
            'case': case,
            'model': res['model_name'],
            'target': target_name,
            'mse_tr': res['mse_tr'],
            'mse_te': res['mse_te'],
            'rmse_te': res['rmse_te'],
            'rel_rmse_te_pct': res['rel_rmse_te_pct'],
        })

    # ---- 6.25  Все 4 модели на датасете i=16
    print('\n' + '=' * 70)
    print('[2.16-all] Все модели для i=16')
    print('=' * 70)
    target_name_16 = 'E(R), В/м'  # 16 % 2 == 0
    print('  Генерация датасета i=16 (1024 строк) ...')
    X16, H0_16, ER_16 = generate_dataset(16, N_euler=300)
    y16 = ER_16
    all_models_i16 = []
    for fc in range(4):
        print(f'  Обучение модели case={fc} ...')
        res16 = train_and_eval(16, X16, y16, target_name_16, feature_names, force_case=fc)
        print(f'  {res16["model_name"]:36s}  RMSE={res16["rmse_te"]:.4e}  ({res16["rel_rmse_te_pct"]:.2f}%)')
        plot_case(res16, os.path.join(FIG_DIR, f'case_i16_c{fc}.png'))
        all_models_i16.append(res16)
    plot_model_comparison(all_models_i16, os.path.join(FIG_DIR, 'case_i16_comparison.png'))
    print(f'  Сравнительный график: figures/case_i16_comparison.png')

    print('\n  Бенчмарк всех моделей для i=16 (RMSE vs N) ...')
    bench_i16 = dataset_size_benchmark_all_models(16, sizes=[64, 128, 256, 512, 1024, 2048], N_euler=150)
    plot_dataset_size_benchmark_all_models(
        16, bench_i16, os.path.join(FIG_DIR, 'dataset_size_benchmark_i16.png'))
    print(f'  График бенчмарка: figures/dataset_size_benchmark_i16.png')

    # ---- 6.3  Финальная сводка
    print('\n' + '=' * 70)
    print('СВОДКА')
    print('=' * 70)
    print(f"{'i':>4}  {'i%4':>4}  {'модель':<36}  {'цель':<12}  "
          f"{'MSE_test':>12}  {'rel_RMSE,%':>11}")
    for s in summary:
        print(f"{s['i_var']:>4}  {s['case']:>4}  {s['model']:<36}  {s['target']:<12}  "
              f"{s['mse_te']:>12.4e}  {s['rel_rmse_te_pct']:>11.3f}")
    print(f'\nГрафики сохранены в: {FIG_DIR}')

    # ---- 6.4  Бенчмарк: RMSE vs размер датасета
    print('\n' + '=' * 70)
    print('[3] Бенчмарк: качество модели vs размер обучающей выборки')
    print('=' * 70)
    bench_sizes = [64, 128, 256, 512, 1024, 2048]
    bench_results = dataset_size_benchmark(sizes=bench_sizes, i_vars=[0, 1, 2, 3, 16], N_euler=150)
    plot_dataset_size_benchmark(
        bench_results, os.path.join(FIG_DIR, 'dataset_size_benchmark.png'))
    plot_dataset_size_benchmark_detail(
        bench_results, os.path.join(FIG_DIR, 'dataset_size_benchmark_detail.png'))
    print(f'\n  Графики бенчмарка сохранены в: {FIG_DIR}')

    # сохраним сводку в текстовый файл
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('Космическая плазма — сводка результатов\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Порядок схемы Эйлера: p_y ≈ {pY:.3f}, p_w ≈ {pW:.3f} (теория: 1)\n\n')
        f.write(f"{'i':>4}  {'i%4':>4}  {'модель':<36}  {'цель':<12}  "
                f"{'MSE_test':>12}  {'rel_RMSE,%':>11}\n")
        for s in summary:
            f.write(f"{s['i_var']:>4}  {s['case']:>4}  {s['model']:<36}  {s['target']:<12}  "
                    f"{s['mse_te']:>12.4e}  {s['rel_rmse_te_pct']:>11.3f}\n")

        f.write('\n\nБенчмарк: RMSE vs размер обучающей выборки\n')
        f.write('=' * 70 + '\n')
        f.write(f"{'i%4':>4}  {'N':>6}  {'RMSE':>14}  {'rel_RMSE,%':>11}\n")
        for i_var, size_res in bench_results.items():
            for r in size_res:
                f.write(f"{i_var:>4}  {r['n']:>6}  {r['rmse']:>14.4e}  {r['rel_pct']:>11.3f}\n")

    return summary, (hs, eY, eW, pY, pW), bench_results


if __name__ == '__main__':
    main()
