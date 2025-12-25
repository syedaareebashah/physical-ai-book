import React, { useState } from 'react';
import { useRouter } from 'next/router';

interface FormData {
  email: string;
  password: string;
  softwareExperience: string;
  programmingLanguages: string;
  devExperienceYears: string;
  hardwareSpecs: {
    deviceType: string;
    os: string;
    cpu: string;
    gpu: string;
    ram: string;
  };
  developmentFocus: string;
}

const SignupForm: React.FC = () => {
  const router = useRouter();
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
    softwareExperience: '',
    programmingLanguages: '',
    devExperienceYears: '',
    hardwareSpecs: {
      deviceType: '',
      os: '',
      cpu: '',
      gpu: '',
      ram: '',
    },
    developmentFocus: '',
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;

    if (name.startsWith('hardwareSpecs.')) {
      const field = name.split('.')[1];
      setFormData({
        ...formData,
        hardwareSpecs: {
          ...formData.hardwareSpecs,
          [field]: value,
        },
      });
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Prepare programming languages as array
      const programmingLanguagesArray = formData.programmingLanguages
        .split(',')
        .map(lang => lang.trim())
        .filter(lang => lang.length > 0);

      // Prepare development focus as array
      const developmentFocusArray = formData.developmentFocus
        .split(',')
        .map(focus => focus.trim())
        .filter(focus => focus.length > 0);

      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          softwareExperience: formData.softwareExperience,
          programmingLanguages: programmingLanguagesArray,
          devExperienceYears: parseInt(formData.devExperienceYears) || 0,
          hardwareSpecs: {
            deviceType: formData.hardwareSpecs.deviceType,
            os: formData.hardwareSpecs.os,
            cpu: formData.hardwareSpecs.cpu,
            gpu: formData.hardwareSpecs.gpu,
            ram: parseInt(formData.hardwareSpecs.ram) || 0,
          },
          developmentFocus: developmentFocusArray,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Redirect to login or dashboard
        router.push('/auth/login');
      } else {
        setError(data.message || 'Signup failed');
      }
    } catch (err) {
      setError('An error occurred during signup');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-center">Create Account</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="email" className="block text-gray-700 mb-2">
            Email
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="password" className="block text-gray-700 mb-2">
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="softwareExperience" className="block text-gray-700 mb-2">
            Software Experience Level
          </label>
          <select
            id="softwareExperience"
            name="softwareExperience"
            value={formData.softwareExperience}
            onChange={handleChange}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select your experience level</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div className="mb-4">
          <label htmlFor="programmingLanguages" className="block text-gray-700 mb-2">
            Programming Languages (comma separated)
          </label>
          <input
            type="text"
            id="programmingLanguages"
            name="programmingLanguages"
            value={formData.programmingLanguages}
            onChange={handleChange}
            placeholder="e.g., JavaScript, Python, Java"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="devExperienceYears" className="block text-gray-700 mb-2">
            Years of Development Experience
          </label>
          <input
            type="number"
            id="devExperienceYears"
            name="devExperienceYears"
            value={formData.devExperienceYears}
            onChange={handleChange}
            min="0"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3">Hardware Specifications</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="hardwareSpecs.deviceType" className="block text-gray-700 mb-2">
                Device Type
              </label>
              <select
                id="hardwareSpecs.deviceType"
                name="hardwareSpecs.deviceType"
                value={formData.hardwareSpecs.deviceType}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select device type</option>
                <option value="laptop">Laptop</option>
                <option value="desktop">Desktop</option>
                <option value="tablet">Tablet</option>
              </select>
            </div>

            <div>
              <label htmlFor="hardwareSpecs.os" className="block text-gray-700 mb-2">
                Operating System
              </label>
              <input
                type="text"
                id="hardwareSpecs.os"
                name="hardwareSpecs.os"
                value={formData.hardwareSpecs.os}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label htmlFor="hardwareSpecs.cpu" className="block text-gray-700 mb-2">
                CPU
              </label>
              <input
                type="text"
                id="hardwareSpecs.cpu"
                name="hardwareSpecs.cpu"
                value={formData.hardwareSpecs.cpu}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label htmlFor="hardwareSpecs.gpu" className="block text-gray-700 mb-2">
                GPU
              </label>
              <input
                type="text"
                id="hardwareSpecs.gpu"
                name="hardwareSpecs.gpu"
                value={formData.hardwareSpecs.gpu}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="md:col-span-2">
              <label htmlFor="hardwareSpecs.ram" className="block text-gray-700 mb-2">
                RAM (GB)
              </label>
              <input
                type="number"
                id="hardwareSpecs.ram"
                name="hardwareSpecs.ram"
                value={formData.hardwareSpecs.ram}
                onChange={handleChange}
                min="0"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        <div className="mb-6">
          <label htmlFor="developmentFocus" className="block text-gray-700 mb-2">
            Development Focus Areas (comma separated)
          </label>
          <input
            type="text"
            id="developmentFocus"
            name="developmentFocus"
            value={formData.developmentFocus}
            onChange={handleChange}
            placeholder="e.g., web, mobile, AI/ML"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 flex items-center justify-center"
        >
          {loading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Creating Account...
            </>
          ) : (
            'Sign Up'
          )}
        </button>
      </form>

      <div className="mt-4 text-center">
        <p className="text-gray-600">
          Already have an account?{' '}
          <a href="/auth/login" className="text-blue-600 hover:underline">
            Sign in
          </a>
        </p>
      </div>
    </div>
  );
};

export default SignupForm;